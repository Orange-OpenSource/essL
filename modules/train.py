# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import math
import logging, os
from typing import List
from datetime import datetime
import torch
from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks import Callback

class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear learning rate scheduler with warmup, hold, and decay. 
    self._step_count is updated once every training epoch and not every
    training step
    
    [1] https://gitlab.tech.orange/shiva-speech/cramming-hubert/-/blob/main/training/lightning.py
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]

class GreedyCTCDecoder(torch.nn.Module):
    """
    https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
    """
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    
class TrainRequeue(Callback):
    def __init__(self, nb_epochs_before_requeue=1, checkpoint=None):
        super().__init__()
        self.first_epoch=None
        self.nb_epochs_before_requeue = nb_epochs_before_requeue
        self.checkpoint = checkpoint
        self.start = datetime.now()

    def on_train_epoch_start(self, trainer, pl_module):
        # current_epoch will be a multiple of nb_epochs_before_requeue when restoring checkpoint
        if not self.first_epoch:
            self.first_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer, outputs, pl_module):
        # https://www.geeksforgeeks.org/how-to-check-the-execution-time-of-python-script/
        td = float((datetime.now() - self.start).total_seconds()) / 3600.0
        if  ("SLURM_JOB_ID" in os.environ and td > 4.0) :
            logging.info(
                "Time threshold, {0} hr, prepare for requeueing".format(td))
            trainer.save_checkpoint(self.checkpoint)
            trainer.save_checkpoint(self.checkpoint.replace('0.ckpt', 
                str(trainer.current_epoch) + '.ckpt'))
            exit(42) # magic exit code for slurm to plan requeueing
