# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import logging, os
import numpy as np
import math
import time
from typing_extensions  import override

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from torchaudio.models.decoder import download_pretrained_files
from torchaudio.models.decoder import ctc_decoder

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from speechbrain.processing.features import (
    STFT, 
    Filterbank,
    spectral_magnitude, 
    InputNormalization 
)
from speechbrain.processing.speech_augmentation import AddNoise

from modules.train import (
    GreedyCTCDecoder,
    TriStageLRScheduler,
    TrainRequeue
)
from modules.external import (
    ConvFeatureExtractionModel, 
    TransformerEncoder,
    ProjUpsampling
)
from modules.perturb import (
    SpecAugment,
    RandomShift,
    RandomNoisePerturbation
)
from modules.simple_wer import SimpleWER
from config.block import (
    BlockConfig,
    ShiftPerturbConfig
) 

class EfficientSSL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.data_dir
        self.checkpoint = os.path.join(config.data_dir, config.checkpoint)
        self.p_checkpoint = os.path.join(config.data_dir, config.p_checkpoint)
        self.freeze_finetuning = config.freeze_finetuning
        self.pretraining = config.pretraining
        self.precision = config.precision
        self.gpus = config.gpus

        self.nb_epochs_before_requeue = config.nb_epochs_before_requeue
        self.pretrain_steps = config.pretrain_steps
        self.finetune_steps = config.finetune_steps
        self.pretrain_lr = config.pretrain_lr
        self.finetune_lr = config.finetune_lr
        self.check_val_every_n_epoch = config.check_val_every_n_epoch
        self.accumulate_grad_batches = config.accumulate_grad_batches 
        self.optimizer_betas = config.optimizer_betas
        self.optimizer_weight_decay = config.optimizer_weight_decay

        self.n_negatives = config.n_negatives 
        self.cross_sample_negatives = config.cross_sample_negatives
        self.logit_temp = config.logit_temp
        
        self.beam_size = config.beam_size
        self.lm_weight = config.lm_weight
        self.word_score = config.word_score

        # Decoders
        self.load_ctc_decoder()
        self.greedy_decoder = GreedyCTCDecoder(self.tokens)

        # Model
        logging.info('Building model ...')
        self.student = self.create_model(is_student=True)
        self.teacher = self.create_model(is_student=False)
        logging.info('Student \n{0}'.format(self.student))
        logging.info('Teacher \n{0}'.format(self.teacher))
        logging.info('Classifier \n{0}'.format(self.classifier))

        self.former_step = -1
        self.val_step_losses = []
        self.val_beam_wers = SimpleWER()
        self.val_greed_wers = SimpleWER()

        # Augmentation 
        self.load_random_shift()

    def create_model(self, is_student: bool) -> torch.nn.Sequential:
        """
        Build the model for student or teacher modules, using Spiral Base 
        architecture as the template

        [1] https://openreview.net/forum?id=TBpg4PnXhYH

        Args:
            is_student: wether to add a convolutional predictor on top 
            of the module, for student only
        Returns:
            nn.Sequential with the components of the model     
        """
        conv1_cfg = BlockConfig()
        conv1_cfg.conv_feature_layers = "[(384, 5, 2), (512, 5, 2),(512, 1, 1)]"
        conv1_cfg.extractor_mode = "layer_norm"
        conv1_cfg.normalize = True
        conv1_cfg.mel_filters = self.config.n_mels
        conv1_cfg.dropout = 0.0
        conv1 = ConvFeatureExtractionModel(
            conv_layers=eval(conv1_cfg.conv_feature_layers),
            dropout=conv1_cfg.dropout,
            input_d= conv1_cfg.mel_filters,
            mode=conv1_cfg.extractor_mode,
            conv_bias=conv1_cfg.conv_bias,
        )

        transf1_cfg = BlockConfig()
        transf1_cfg.encoder_layers = 2 
        transf1_cfg.encoder_embed_dim = 512
        transf1_cfg.encoder_ffn_embed_dim = 2048
        transf1_cfg.encoder_layerdrop = 0
        transf1_cfg.encoder_attention_heads = 8
        transf1_cfg.relative_position_embedding = True
        transf1_cfg.gru_rel_pos = True
        transf1 = TransformerEncoder(transf1_cfg)

        conv2_cfg = BlockConfig()
        conv2_cfg.conv_feature_layers ="[(1536, 5, 2), (768, 1, 1)]"
        conv2_cfg.extractor_mode = "layer_norm"
        conv2_cfg.normalize = True
        conv2_cfg.dropout = 0.0
        conv2 = ConvFeatureExtractionModel(
            conv_layers=eval(conv2_cfg.conv_feature_layers),
            dropout=conv2_cfg.dropout,
            input_d=transf1_cfg.encoder_embed_dim,
            mode=conv2_cfg.extractor_mode,
            conv_bias=conv2_cfg.conv_bias,
        )

        transf2_cfg = BlockConfig()
        transf2_cfg.encoder_layers = 10 
        transf2_cfg.encoder_embed_dim = 768
        transf2_cfg.encoder_ffn_embed_dim = 3072
        transf2_cfg.encoder_layerdrop = 0.05
        transf2_cfg.encoder_attention_heads = 12
        transf2_cfg.relative_position_embedding = True
        transf2_cfg.gru_rel_pos = True
        transf2_cfg.extract_layer_features = is_student
        transf2 = TransformerEncoder(transf2_cfg)

        projection_head = nn.Linear(
            in_features=transf2_cfg.encoder_embed_dim, out_features=256)

        model = nn.Sequential(
            conv1, transf1, conv2, transf2, projection_head)
        
        if is_student == True:
            self.avg_weights = torch.rand(
                transf2_cfg.encoder_layers, requires_grad=True)
            
            predictor_cfg = BlockConfig()
            predictor_cfg.conv_feature_layers = \
                "[(256, 5, 1), (256, 5, 1), (256, 1, 1)]"
            predictor_cfg.extractor_mode = "layer_norm"
            predictor_cfg.normalize = True
            predictor_cfg.conv_padding = "same"
            predictor_cfg.conv_bias = 0.0
            self.predictor = ConvFeatureExtractionModel(
                conv_layers=eval(predictor_cfg.conv_feature_layers),
                dropout=predictor_cfg.dropout,
                input_d=256,
                mode=predictor_cfg.extractor_mode,
                padding=predictor_cfg.conv_padding,
                conv_bias=predictor_cfg.conv_bias,
                )
            
            classifier_cfg = BlockConfig()
            classifier_cfg.conv_feature_layers = \
                "[(512, 5, 1), (512, 5, 1)]"
            classifier_cfg.extractor_mode = "default"
            classifier_cfg.normalize = False
            classifier_cfg.conv_bias = 0.0
            self.classifier = nn.Sequential(
                ProjUpsampling(
                    in_channels=256,
                    rate=4, 
                    filters=512, 
                    kernel_size=(5,),
                    norm_type='ln', 
                    act_func='relu', 
                    dropout=classifier_cfg.dropout),
                ConvFeatureExtractionModel(
                    conv_layers=eval(classifier_cfg.conv_feature_layers),
                    dropout=classifier_cfg.dropout,
                    input_d=512,
                    mode=classifier_cfg.extractor_mode,
                    conv_bias=classifier_cfg.conv_bias,
                    ),
                nn.Linear(
                    in_features=512, out_features=len(self.tokens))
                )
            
        return model
    
    def load_ctc_decoder(self):
        """
        Load CTC decoder using librispeech 4-gram language model

        https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        """
        torch.hub.set_dir(self.data_dir)
        lm_files = download_pretrained_files("librispeech-4-gram")
        self.tokens = []
        self.token_class = {}
        with open(lm_files.tokens) as tokens_file:
            lines = tokens_file.readlines()
            num_tokens = 0
            for line in lines:
                line = line.strip()
                self.tokens.append(line)
                self.token_class[line] = num_tokens
                num_tokens += 1
        logging.info("Tokens %s" % self.tokens)
        logging.info("Token class %s" % self.token_class)

        self.beam_search_decoder = ctc_decoder(
            lexicon=lm_files.lexicon,
            tokens=lm_files.tokens,
            lm=lm_files.lm,
            nbest=3,
            beam_size=self.beam_size,
            lm_weight=self.lm_weight,
            word_score=self.word_score,
            )
    
    def load_random_shift(self):
        """
        Add positional random shift to avoid the student to learn positional 
        from input poistional information 

        [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/examples/asr/conf/spiral/spiral_base_pretrain_ls960_noise.py
        """
        shift_config = ShiftPerturbConfig(
            dist='uniform',
            shift_prob=1.0,
            max_ratio=0.5,
            unit=8,
            max=16,
            min=0,
            truncate=False)
        self.random_shift = RandomShift(shift_config)

    def forward(self, x):
        x_student = x
        y_student = self.student(x_student)
        y_teacher = self.teacher(x_student)
        logging.debug("Fwd student output shape {0}".format(y_student.shape))
        logging.debug("Device {0}".format(y_student.device))
        logging.debug("Type {0}".format(y_student.dtype))
        logging.debug("Fwd Teacher output shape {0}".format(y_teacher.shape))
        logging.debug("Device {0}".format(y_teacher.device))
        logging.debug("Type {0}".format(y_teacher.dtype))
        y_student = self.predictor(y_student)
        y_student = self.classifier(y_student)
        logging.debug("classifier output shape {0}".format(y_student.shape))
        return y_student

    def validation_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, "Validation")

    def test_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, "Test")

    def on_validation_epoch_end(self):
        self.shared_eval_epoch_end("Validation")

    def on_test_epoch_end(self):
        self.shared_eval_epoch_end("Test")

    def shared_eval_step(self, batch, batch_idx, test_type):
        signal  = batch.signal.data
        words   = batch.words # List with sentences for audio sequences
        logging.debug("Signal shape {0}".format(signal.shape))

        targets = []
        target_lengths = []
        for i in range(len(words)):
            sentence = words[i]
            target_lengths.append(len(sentence))
            for char in sentence:
                targets.append(self.token_class.get(char,self.token_class['|']))

        x_student = signal
        x_student = x_student.transpose(1, 2) # BxTxC to BxCxT
        if self.precision == 16:
            x_student = x_student.to(torch.float16)
        x_student = x_student.transpose(1, 2) # BxCxT to BxTxC
        y_student = self.student(x_student)
        logging.debug("Log mel x_student shape {0}".format(x_student.shape))
        logging.debug("y_student shape {0}".format(y_student.shape))

        y_student = self.predictor(y_student)
        logging.debug("y_student shape {0}".format(y_student.shape))
        y_student = self.classifier(y_student)
        logging.debug("Classifier output shape {0}".format(y_student.shape))

        T = y_student.shape[1] # Sequence length 
        N = y_student.shape[0] # Batch size
        targets = torch.as_tensor(
            targets, dtype=torch.long, device=y_student.device)
        target_lengths = torch.as_tensor(
            target_lengths, dtype=torch.long, device=y_student.device)
        input_lengths = torch.full(
            size=(N,), fill_value=T, dtype=torch.long, device=y_student.device)
        
        # Convert from B x T x C to T x B x C
        y_student = y_student.transpose(0, 1)
        ctc_loss = torch.nn.CTCLoss(
            blank=0, reduction='none', zero_infinity=False)
        log_softmax = torch.nn.LogSoftmax(dim=2)
        log_probs = log_softmax(y_student)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss = torch.mean(loss)

        logging.debug("CTC targets shape {0}".format(targets.shape))
        logging.debug("CTC target lengths {0}".format(target_lengths))
        logging.debug("CTC input lengths {0}".format(input_lengths))
        logging.debug("CTC log_probs shape {0}".format(log_probs.shape))

        logging.info((test_type + ", " +
            "batch_loss = {0} step = {1} batch id = {2} epoch = {3}").format(
            loss, self.global_step, batch_idx, self.current_epoch))
        
        # https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        # Convert from T x B x C to B x T x C
        y_student = y_student.transpose(0, 1).float()
        emissions = log_softmax(y_student)
        emissions = emissions.cpu()

        for i in range(len(words)):
            actual_transcript = words[i]
            emission = emissions[i]
            emission = emission.unsqueeze(0)

            greedy_result = self.greedy_decoder(emission[0])
            greedy_transcript = " ".join(greedy_result).strip()
            self.val_greed_wers.AddHypRef(
                greedy_transcript, actual_transcript)
           
            beam_search_result = self.beam_search_decoder(emission)
            beam_search_transcript = " ".join(
                beam_search_result[0][0].words).strip()
            self.val_beam_wers.AddHypRef(
                beam_search_transcript, actual_transcript)
           
            logging.info("Actual transcript: {0}".format(actual_transcript))
            logging.info("Greedy: {0}".format(greedy_transcript))
            logging.info("Beam: {0}".format(beam_search_transcript))

        self.val_step_losses.append(loss)
        return loss

    def shared_eval_epoch_end(self, test_type):
        loss = sum(self.val_step_losses) / len(self.val_step_losses)
        greedy = self.val_greed_wers.GetWER()
        beam = self.val_beam_wers.GetWER()
        logging.info((test_type + ", " +
            "loss = {0} greedy = {1} beam = {2} step = {3} epoch = {4}").format(
            loss, greedy, beam, self.global_step, self.current_epoch))
        
        self.val_step_losses.clear()  # free memory
        self.val_beam_wers = SimpleWER()
        self.val_greed_wers = SimpleWER()

    def pretrain_model(self, batch, batch_idx):
        """
        https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/st2vec/st2vec_model.py
        """
        signal  = batch.signal.data
        signal_noise  = batch.signal_noise.data

        logging.debug("Signal shape {0}".format(signal.shape))
        logging.debug("Signal noise shape {0}".format(signal_noise.shape))

        x_teacher = signal
        x_student = signal_noise
        x_student = x_student.transpose(1, 2) # BxTxC to BxCxT
        if self.precision == 16:
            x_student = x_student.to(torch.float16)
        B, C, T = x_student.shape
        x_student_lengths = torch.full(
            size=(B,), fill_value=T, dtype=torch.long, device=x_student.device)
        self.specaug = SpecAugment(
            freq_masks=int(0.020 * C), freq_width=20,
            time_masks=int(0.025 * T), time_width=20,
            max_time_masks=100, gauss_mask_std=1.0)
        x_student = self.specaug(x_student, x_student_lengths)
        x_student = x_student.transpose(1, 2) # BxCxT to BxTxC
        y_student = self.student(x_student)
        y_student = self.predictor(y_student)

        x_teacher = x_teacher.transpose(1, 2) # BxTxC to BxCxT
        if self.precision == 16:
            x_teacher = x_teacher.to(torch.float16)
        x_teacher = x_teacher.transpose(1, 2) # BxCxT to BxTxC
        B, T, C = x_teacher.shape
        x_teacher_lengths = torch.full(
            size=(B,), fill_value=T, 
            dtype=torch.long, device=x_teacher.device)
        x_teacher, x_teacher_lengths, teacher_shift_num, _, \
            teacher_r_shift_num = \
            self.random_shift.shift(x_teacher, x_teacher_lengths, 0.0)

        # Update teacher from student weights and calculate targets, skipping 
        # inner steps when using grad accumulation
        if self.global_step == 0:
            target_momentum = 0
            self.ema_update(self.teacher, self.student, target_momentum)
        elif self.global_step != self.former_step: 
            target_momentum = self.momentum_schedule(
                base_value=0.995, final_value=1.0, 
                max_steps=self.pretrain_steps, step=self.global_step, 
                type='cosine')
            self.ema_update(self.teacher, self.student, target_momentum)
            self.former_step = self.global_step
        with torch.no_grad():
            y_teacher = self.teacher(x_teacher)

        logging.debug("Log mel x_teacher shape {0}".format(x_teacher.shape))
        logging.debug("Log mel x_student shape {0}".format(x_student.shape))
        logging.debug("y_student shape {0}".format(y_student.shape))
        logging.debug("y_teacher shape {0}".format(y_teacher.shape))

        # Readjust tensors to account for random shifting
        B, T, C = y_teacher.shape
        y_teacher_lengths = torch.full(
            size=(B,), fill_value=T, dtype=torch.long, device=y_teacher.device)
        if teacher_shift_num > 0:
            y_teacher = y_teacher[:, teacher_shift_num:]
            y_teacher_lengths = y_teacher_lengths - teacher_shift_num
        else:
            assert teacher_shift_num == 0
        if teacher_r_shift_num > 0:
            y_teacher = y_teacher[:, :-teacher_r_shift_num]
            y_teacher_lengths = y_teacher_lengths - teacher_r_shift_num
        else:
            assert teacher_r_shift_num == 0

        assert y_student.shape[1] == y_teacher.shape[1]
        
        # Calculate contrastive loss
        T = y_teacher.shape[1] # Sequence length 
        N = y_teacher.shape[0] # Batch size
        y_teacher_lengths = torch.full(
            size=(N,), fill_value=T, dtype=torch.long, device=y_teacher.device)
        y_teacher = y_teacher.contiguous()
        y_student = y_student.contiguous()
        y_teacher = y_teacher.view(1, -1, y_teacher.size(-1))
        y_student = y_student.view(1, -1, y_student.size(-1))
        sampled_negatives, _ = self.sample_negatives_flat(
            y_teacher, y_teacher_lengths.tolist())
        loss, accuracy = self.contrastive_loss(
            y_student, y_teacher, sampled_negatives)
        
        logging.debug("y_student shape {0}".format(y_student.shape))
        logging.debug("y_teacher shape {0}".format(y_teacher.shape))
        logging.debug("negatives shape {0}".format(sampled_negatives.shape))
        logging.info(("Pretraining, pytorch loss = {0} step = {1} " +
            "batch id = {2} epoch = {3} lr = {4} accuracy = {5}").format(
            loss, self.global_step, batch_idx, self.current_epoch, 
            self.optimizers().param_groups[0]['lr'], accuracy))
        self.log('pt_loss', loss)
        self.log('pt_acc',  accuracy)
        self.log('pt_lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def finetune_model(self, batch, batch_idx):
        #https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
        #https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
        #https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        #https://distill.pub/2017/ctc/
        signal  = batch.signal.data
        words   = batch.words # List with sentences for audio sequences
        lengths = batch.length # List with audio sequences duration in seconds
        logging.debug("Signal shape {0}".format(signal.shape))

        targets = []
        target_lengths = []
        for i in range(len(words)):
            sentence = words[i]
            target_lengths.append(len(sentence))
            for char in sentence:
                targets.append(self.token_class.get(char,self.token_class['|']))

        if self.freeze_finetuning == True:
            logging.info("Freezing backbone model ...")
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.predictor.parameters():
                p.requires_grad = False
            for p in self.teacher.parameters():
                p.requires_grad = False

        x_student = signal
        x_student = x_student.transpose(1, 2) # BxTxC to BxCxT
        if self.precision == 16:
            x_student = x_student.to(torch.float16)
        B, C, T = x_student.shape
        x_student_lengths = torch.full(
            size=(B,), fill_value=T, dtype=torch.long, device=x_student.device)
        self.specaug = SpecAugment(
            freq_masks=int(0.020 * C), freq_width=20,
            time_masks=int(0.025 * T), time_width=20,
            max_time_masks=100, gauss_mask_std=1.0)
        x_student = self.specaug(x_student, x_student_lengths)
        x_student = x_student.transpose(1, 2) # BxCxT to BxTxC
        y_student = self.student(x_student)
        logging.debug("Log mel x_student shape {0}".format(x_student.shape))
        logging.debug("y_student shape {0}".format(y_student.shape))

        y_student = self.predictor(y_student)
        logging.debug("y_student shape {0}".format(y_student.shape))
        y_student = self.classifier(y_student)
        logging.debug("Classifier output shape {0}".format(y_student.shape))

        T = y_student.shape[1] # Sequence length 
        N = y_student.shape[0] # Batch size
        targets = torch.as_tensor(
            targets, dtype=torch.long, device=y_student.device)
        target_lengths = torch.as_tensor(
            target_lengths, dtype=torch.long, device=y_student.device)
        input_lengths = torch.full(
            size=(N,), fill_value=T, dtype=torch.long, device=y_student.device)
        logging.debug("CTC targets shape {0}".format(targets.shape))
        logging.debug(
            "CTC target lengths shape {0}".format(target_lengths.shape))
        logging.debug(
            "CTC input lengths shape {0}".format(input_lengths.shape))

        # Convert from B x T x C to T x B x C
        y_student = y_student.transpose(0, 1)
        ctc_loss = torch.nn.CTCLoss(
            blank=0, reduction='none', zero_infinity=False)
        log_softmax = torch.nn.LogSoftmax(dim=2)
        log_probs = log_softmax(y_student)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss = torch.mean(loss)

        logging.info(("Fine tuning, pytorch loss = {0}" +
            " step = {1} batch id = {2} epoch = {3} lr = {4}").format(
            loss, self.global_step, batch_idx, self.current_epoch, 
            self.optimizers().param_groups[0]['lr']))
        self.log('ft_loss', loss)
        self.log('ft_lr', self.optimizers().param_groups[0]['lr']) 
        return loss
    
    def pretrain_optimizers(self):
        """
        [1] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        [2] https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
        [3] https://github.com/Lightning-AI/lightning/issues/3051
        """
        logging.info("Configuring pretrain optimizers ...")
        adam = torch.optim.AdamW(
            self.parameters(), 
            lr=self.pretrain_lr, 
            betas=self.optimizer_betas, 
            weight_decay=self.optimizer_weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer=adam, max_lr=self.pretrain_lr,
                total_steps=self.pretrain_steps,
                pct_start=0.08, anneal_strategy='cos',
                div_factor=1e6, final_div_factor=1,
                ),
            'interval': 'step',
            'frequency': 1
            }
        return {'optimizer' : adam, 'lr_scheduler' : scheduler}
    
    def finetune_optimizers(self):
        logging.info("Configuring finetune optimizers ...")
        adam = torch.optim.AdamW(
            self.parameters(), 
            lr=self.finetune_lr, 
            betas=self.optimizer_betas, 
            weight_decay=self.optimizer_weight_decay)
        scheduler = {
            'scheduler' : TriStageLRScheduler(
                optimizer=adam, init_lr_scale=1e-6, final_lr_scale=1e-6,
                warmup_updates=0.1 * self.finetune_steps,
                hold_updates=0.4 * self.finetune_steps,
                decay_updates=0.5 * self.finetune_steps,
                ),
            'interval': 'step',
            'frequency': 1
            }
        return {'optimizer' : adam, 'lr_scheduler' : scheduler}
    
    def train_model(
        self, pretrain_dataloader, finetune_dataloader, val_dataloader):
        """
        Use Pytorch Lightning to train the model

        Args
            pretrain_dataloader: data for self-supervised pretraining
            finetune_dataloader: data for supervised finetuning
            val_dataloader: data for validating the module
        Returns:
        [1] https://pytorch-lightning.readthedocs.io/en/1.1.0/trainer.html
        """
        if self.pretraining == True:
            logging.info('Pretraining model ...')
            self.steps_per_epoch = len(pretrain_dataloader)
            self.configure_optimizers = self.pretrain_optimizers
            self.training_step = self.pretrain_model
            logger = TensorBoardLogger('lightning_logs')
            trainer = pl.Trainer(
                gpus=self.gpus, precision=self.precision,
                max_steps=self.pretrain_steps, logger=logger,
                accumulate_grad_batches=self.accumulate_grad_batches)
            trainer.fit(self, 
                train_dataloader=pretrain_dataloader)
            trainer.save_checkpoint(self.p_checkpoint)

        logging.info('Finetuning model ...')
        self.steps_per_epoch = len(finetune_dataloader)
        self.configure_optimizers = self.finetune_optimizers
        self.training_step = self.finetune_model
        logger = TensorBoardLogger('lightning_logs')
        trainer = pl.Trainer(
            gpus=self.gpus, precision=self.precision,
            max_steps=self.finetune_steps, logger=logger,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_grad_batches)
        trainer.fit(self, 
            train_dataloader=finetune_dataloader,
            val_dataloaders=val_dataloader)
        trainer.save_checkpoint(self.checkpoint)
        
    def test_model(self, test_dataloader):
        """
        Use Pytorch Lightning to load model from checkpoint and test it
        Args
            test_dataloader: data for testing the module. It can be a list of
            datasets
        Returns:

        [1] https://pytorch-lightning.readthedocs.io/en/1.1.0/introduction_guide.html
        [2] https://pytorch-lightning.readthedocs.io/en/1.1.0/trainer.html
        """
        config = self.config
        self = self.load_from_checkpoint(self.checkpoint, config=config)
        trainer = pl.Trainer(gpus=self.gpus, precision=self.precision) 
        trainer.test(
            self, test_dataloaders=test_dataloader, ckpt_path=self.checkpoint)

    def sample_negatives_flat(self, y, nums):
        """
        Negatives to calculate the contrastive loss
        usage: sampled_negatives, _ = self.sample_negatives(targets, targets.size(1))

        [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/wav2vec/wav2vec_model.py
        """
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        assert bsz == 1 and tsz == sum(nums)  # fake batch dim
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # cross_high = tsz * bsz

        neg_idxs_l = []
        idx_start = 0
        with torch.no_grad():
            for i, num_i in enumerate(nums):
                assert num_i > 1, f"{bsz, tsz, fsz}"

                assert self.n_negatives > 0
                tszs_i = buffered_arange(
                    num_i).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

                high_i = num_i
                neg_idxs_i = torch.randint(
                    low=0, high=high_i - 1, size=(self.n_negatives * num_i,))
                neg_idxs_i[neg_idxs_i >= tszs_i] += 1

                neg_idxs_i += idx_start
                idx_start += num_i

                neg_idxs_l.append(neg_idxs_i)

                assert self.cross_sample_negatives == 0

        neg_idxs = torch.cat(neg_idxs_l)
        assert neg_idxs.ndim == 1

        negs = y[neg_idxs]
        negs = negs.view(
            bsz, sum(nums), 
            self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3)  # to NxBxTxC
        return negs, neg_idxs
    
    def ema_update(self, ema_module, new_module, m):
        with torch.no_grad():
            for param_q, param_k in zip(
                new_module.parameters(), ema_module.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def momentum_schedule(self, base_value, final_value, max_steps, step, type):
        if type == 'linear':
            if step <= max_steps:
                cur_value = base_value + (final_value - base_value) * (step / max_steps)
            else:
                cur_value = final_value
            return cur_value
        elif type == 'cosine':
            if step <= max_steps:
                cur_value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * step / max_steps))
            else:
                cur_value = final_value
            return cur_value
        else:
            raise ValueError('unknown scheduler type: {}'.format(type))
    
    def contrastive_loss(
        self,
        logits: torch.tensor,
        targets: torch.tensor,
        negatives: torch.tensor,
        ):
        """
        Args:
            logits: Model activations
            targets: The true target representations
            negatives: Sampled negatives from the input
        
        Returns:
            output loss values

        [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/losses/wav2vecloss.py
        """

        # Calculate similarity between logits and all targets, returning FxBxT
        similarity_scores = self._calculate_similarity(
            logits, negatives, targets)

        # Create targets of size B*T
        similarity_targets = logits.new_zeros(
            similarity_scores.size(1) * similarity_scores.size(2), 
            dtype=torch.long)

        # Transpose similarity scores to (T*B)xF for loss
        similarity_scores = similarity_scores.transpose(0, 2)
        similarity_scores = similarity_scores.reshape(
            -1, similarity_scores.size(-1))

        contrastive_loss = F.cross_entropy(
            similarity_scores, similarity_targets, reduction='mean')
        
        accuracy = None
        with torch.no_grad():
            if similarity_scores.numel() == 0:
                corr = 0
                count = 0
                accuracy = float('nan')
            else:
                assert similarity_scores.dim() > 1, similarity_scores.shape
                max = similarity_scores.argmax(-1) == 0
                min = similarity_scores.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = float(max.numel())
                accuracy = corr / count

        return contrastive_loss, accuracy

    def _calculate_similarity(self, logits, negatives, targets):
        neg_is_pos = (targets == negatives).all(-1)
        targets = targets.unsqueeze(0)
        targets = torch.cat([targets, negatives], dim=0)
        logits = torch.cosine_similarity(
            logits.float(), targets.float(), dim=-1).type_as(logits)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        return logits

# https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL
def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]