# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

"""
Default configuration values for the transformer and convolutional blocks

[1] https://github.com/microsoft/unilm/blob/master/wavlm/WavLM.py
[2] https://openreview.net/forum?id=TBpg4PnXhYH
"""
class BlockConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     
        self.mel_filters = 128

        self.encoder_layers: int = 12  # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072   # dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False  # apply layernorm first 
        self.extract_layer_features: bool = False

        # string describing convolutional feature extraction layers in the 
        # form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_feature_layers: str = \
            "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     
        self.conv_bias: bool = False    # include bias in conv encoder
        self.conv_padding: bool = 'valid'
        self.feature_grad_mult: float = 1.0   # multiply var grads by this

        self.normalize: bool = False  # input to have 0 mean and unit variance

        # dropouts
        self.dropout: float = 0.1   # dropout probability for the transformer
        self.attention_dropout: float = 0.1  #  attention weights
        self.activation_dropout: float = 0.0  # after activation in FFN
        self.encoder_layerdrop: float = 0.0   #  dropping a tarnsformer layer
        self.dropout_input: float = 0.0   # dropout to apply to the input 
        self.dropout_features: float = 0.0  # dropout to apply to the features 

        # positional embeddings
        self.conv_pos: int = 128     # filters for conv positional embeddings
        self.conv_pos_groups: int = 16   # groups for conv positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False  # apply it
        self.num_buckets: int = 320   # number of buckets 
        self.max_distance: int = 1280     # maximum distance 
        self.gru_rel_pos: bool = False     # apply gated 

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

"""
Configurations for audio perturbations

[1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/st2vec/st2vec_config.py
[2] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/spec2vec/spec2vec_config.py
"""
from typing import Optional, List, Any
from dataclasses import field, dataclass
from omegaconf import MISSING

@dataclass
class ShiftPerturbConfig:
    dist: str = 'uniform'
    shift_prob: float = MISSING
    max_ratio: float = 0.5
    unit: int = MISSING
    max: Optional[int] = None
    min: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    truncate: bool = True

@dataclass
class NoisePerturbConfig:
    manifest_path: List[str]
    min_snr_db: float
    max_snr_db: float
    max_gain_db: float = 300.0
    ratio: float = 1.0
    target_sr: int = 16000
    data_dir: str = ''
    cache_noise: bool = False
