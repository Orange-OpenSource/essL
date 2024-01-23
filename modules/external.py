# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import math
import logging
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F

from flash_attn.flash_attention import FlashMHA

from modules.WavLM_modules import (
    get_activation_fn, 
    init_bert_params,
    Fp32GroupNorm, 
    Fp32LayerNorm, 
    TransposeLast, 
    GLU_Linear, 
    SamePad,
    MultiheadAttention,
)

class ConvFeatureExtractionModel(nn.Module):
    """
    Build subsampling convolutional module. Use ReLU activation and 
    B T C for model output

    [1] https://github.com/microsoft/unilm/blob/master/wavlm/WavLM.py
    """
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            input_d: int,
            dropout: float = 0.0,
            mode: str = "default",
            padding: str = "valid",
            conv_bias: bool = False,
            conv_type: str = "default"
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, 
                                 bias=conv_bias, padding=padding)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.ReLU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.ReLU(),
                )
            else:
                return nn.Sequential(
                    make_conv(), nn.Dropout(p=dropout), nn.ReLU())

        self.input_d = input_d
        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = self.input_d
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = self.input_d
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=padding)
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            in_d = self.input_d
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(
                    torch.nn.LayerNorm([dim, idim])
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):
        if len(x.shape) == 2:
            # BxT -> BxCxT 
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)

        # Change output to B x T x C format, convert back to half precision
        # after Fp32LayerNorm or Fp32GroupNorm 
        x = x.transpose(1, 2).contiguous()
        x = x.to(torch.float16)
        logging.debug(
            "Conv block output shape {0} device {1} type {2}".format(
            x.shape, x.device, x.dtype))
        return x
    
class TransformerEncoder(nn.Module):
    """
    Implement a transformer using transformer encoder layers, with 
    flash attention to improve efficiency. Return no layer results in forward
    method.

    [1] https://github.com/microsoft/unilm/blob/master/wavlm/WavLM.py
    [2] https://github.com/HazyResearch/flash-attention
    """
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt(
            (4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(
            self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(
            self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList([
            TransformerEncoderFlashMHA(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                # Comment out for Flash attention only:
                #has_relative_attention_bias=(
                #    self.relative_position_embedding and i == 0),
                #num_buckets=self.num_buckets,
                #max_distance=self.max_distance,
                #gru_rel_pos=args.gru_rel_pos,
            )
            for i in range(args.encoder_layers)
        ])

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)  
        
        logging.debug(
            "Transformer block output shape {0} device {1} type {2}".format(
            x.shape, x.device, x.dtype))
        return x # No layer_results

    def extract_features(
        self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, _ = layer(
                    x, self_attn_padding_mask=padding_mask, 
                    need_weights=False
                    )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        return x, layer_results
    
class TransformerEncoderFlashMHA(nn.Module):
    """
    Implements a Transformer Encoder Layer using flash attention

    [1] https://github.com/microsoft/unilm/blob/master/wavlm/WavLM.py
    [2] https://github.com/HazyResearch/flash-attention
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = FlashMHA(
            self.embedding_dim,
            num_attention_heads,
            attention_dropout=attention_dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(
                self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                x=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                x=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, None

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.

    [1] https://github.com/microsoft/unilm/blob/master/wavlm/WavLM.py
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
            has_relative_attention_bias: bool = False,
            num_buckets: int = 0,
            max_distance: int = 0,
            rescale_init: bool = False,
            gru_rel_pos: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias
    
"""
Convolutional upsampling from Spiral implementation [1], adapting tensor shape 
and module output to EfficientSSL 

[1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/parts/convolution_layers.py
[2] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/examples/asr/conf/spiral/spiral_base_finetune_ls100_char.py
"""

conv_dic = {'1d': torch.nn.Conv1d, '2d': torch.nn.Conv2d}
act_dic = {"hardtanh": nn.Hardtanh, "relu": nn.ReLU}

class ProjUpsampling(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, *, rate, norm_type, act_func,
                 dropout=0.0,
                 padding='same',
                 use_tf_pad=True,
                 ln_eps=1e-5,
                 bias=True):
        super(ProjUpsampling, self).__init__()

        self.upsample_rate = rate
        self.filters = filters
        self.proj = ConvNormAct(in_channels=in_channels, filters=self.filters * self.upsample_rate, kernel_size=kernel_size,
                                stride=(1,), dilation=(1,), norm_type=None, act_func=None,
                                conv_type='1d', dropout=0.0,
                                padding=padding, use_tf_pad=use_tf_pad, ln_eps=ln_eps, gn_groups=None, bias=bias)

        assert norm_type is None or norm_type == 'ln'
        self.norm = get_norm(norm_type, '1d', self.filters, ln_eps=ln_eps, gn_groups=None)
        self.norm_type = norm_type
        self.act = identity if act_func is None else act_dic[act_func]()
        self.drop = identity if dropout == 0 else nn.Dropout(p=dropout)

    def forward(self, x):
        # Convert from B x T x C to B x C x T
        x = x.transpose(1, 2)
        # T to be upsampled after the convolution
        B, C, T = x.size() 
        lens = torch.full(size=(B,), fill_value=T, device=x.device)
        pad_mask = create_pad_mask(lens, max_len=x.size(2))
        output, lens, _ = self.proj(x, lens, pad_mask=pad_mask)
        # Convert back to B x T x C
        output = output.transpose(1, 2)
        B, T, C = output.size()
        output = output.reshape(B, T * self.upsample_rate, self.filters)
        lens = lens * self.upsample_rate
        output = self.norm(output)
        output = self.act(output)
        output = self.drop(output)
        return output

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride, dilation, norm_type, act_func,
                 conv_type,
                 dropout=0.0,
                 padding='same',
                 use_tf_pad=True,
                 ln_eps=1e-5,
                 gn_groups=None,
                 bias=None):
        super(ConvNormAct, self).__init__()

        if bias is None:
            bias = norm_type is None

        self.conv = Conv(in_channels, filters, tuple(kernel_size),
                         stride=tuple(stride),
                         padding=padding,
                         dilation=tuple(dilation),
                         bias=bias,
                         conv_type=conv_type,
                         use_tf_pad=use_tf_pad)
        self.proj_conv = None
        assert conv_type in ['1d', '2d']
        self.norm = get_norm(norm_type, conv_type, filters, ln_eps=ln_eps, gn_groups=gn_groups)
        self.norm_type = norm_type
        self.act = identity if act_func is None else act_dic[act_func]()
        self.drop = identity if dropout == 0 else nn.Dropout(p=dropout)

    def forward(self, x, lens, pad_mask=None):
        # x: [B, C, T] or [B, C, T, F]

        output, lens, pad_mask = self.conv(x, lens, pad_mask)
        if self.norm_type == 'ln':
            output = torch.transpose(output, -1, -2)
        output = self.norm(output)
        if self.norm_type == 'ln':
            output = torch.transpose(output, -1, -2)
        output = self.act(output)
        output = self.drop(output)

        return output, lens, pad_mask

    def update_out_seq_lens(self, lens):
        return self.conv.update_out_seq_lens(lens)

def get_norm(norm_type, conv_type, filters, ln_eps=1e-5, gn_groups=None):
    if norm_type == 'bn':
        if conv_type == '2d':
            norm = nn.BatchNorm2d(filters, momentum=0.01, eps=1e-3)
        else:
            norm = nn.BatchNorm1d(filters, momentum=0.01, eps=1e-3)
    elif norm_type == 'ln':
        assert conv_type != '2d'
        norm = nn.LayerNorm(filters, eps=ln_eps)
    elif norm_type == 'gn':
        assert gn_groups is not None
        norm = nn.GroupNorm(gn_groups, filters)
    else:
        assert norm_type is None, norm_type
        norm = identity
    return norm

# conv wrapper supports same padding, tf style padding, track length change during subsampling
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1,),
                 padding='same',
                 dilation=(1,),
                 bias=True,
                 conv_type='1d',
                 use_tf_pad=False):
        super(Conv, self).__init__()

        self.conv_type = conv_type
        self.is_2d_conv = self.conv_type == '2d'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
            if self.use_conv2d:
                kernel_size = kernel_size * 2

        if isinstance(stride, int):
            stride = (stride,)
            if self.use_conv2d:
                stride = stride * 2

        self.padding = padding

        assert dilation == (1,) or dilation == (1, 1)

        assert use_tf_pad
        self.use_tf_pad = use_tf_pad
        if self.use_tf_pad:
            self.pad_num, self.even_pad_num = get_tf_pad(kernel_size, stride)

        self.conv = conv_dic[self.conv_type](in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=self.get_padding_num(kernel_size, stride, dilation),
                                             bias=bias)
        self.need_pad = kernel_size[0] > 1 or (len(kernel_size) == 2 and kernel_size[1] > 1)
        self.need_pad_mask = kernel_size[0] > 1
        assert stride[0] >= 1
        self.subsample_factor = stride[0]

    def forward(self, x, lens, pad_mask=None):
        # x: [B, C, T] or [B, C, T, F]
        if pad_mask is not None and self.need_pad_mask:
            if self.is_2d_conv:
                x = x.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            else:
                x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        if self.use_tf_pad and self.need_pad:
            x = self.pad_like_tf(x)

        output = self.conv(x)

        if self.subsample_factor > 1:
            lens = self.update_out_seq_lens(lens)
            pad_mask = create_pad_mask(lens, max_len=output.size(2))

        return output, lens, pad_mask

    def get_padding_num(self, kernel_size, stride, dilation):
        if self.padding == 'same':
            if self.use_tf_pad:
                padding_val = 0
            else:
                assert not self.use_tf_pad
                padding_val = get_same_padding(kernel_size, stride, dilation)
        else:
            raise ValueError("currently only 'same' padding is supported")
        return padding_val

    def update_out_seq_lens(self, lens):
        t = 0  # axis of time dimension
        if self.padding == 'same':
            if self.use_tf_pad:
                lens = (lens + self.conv.stride[t] - 1) // self.conv.stride[t]
            else:
                # todo: verify this in pytorch
                lens = (lens + 2 * self.conv.padding[t] - self.conv.dilation[t] * (self.conv.kernel_size[t] - 1) - 1) // self.conv.stride[t] + 1
        else:
            assert self.padding == 'valid' and self.use_tf_pad
            lens = (lens - self.conv.kernel_size[t] + self.conv.stride[t]) // self.conv.stride[t]
        return lens

    def pad_like_tf(self, x):
        if self.is_2d_conv:
            if x.size(-1) % 2 == 0:
                w_pad_num = self.even_pad_num[1]
            else:
                w_pad_num = self.pad_num[1]
            if x.size(-2) % 2 == 0:
                h_pad_num = self.even_pad_num[0]
            else:
                h_pad_num = self.pad_num[0]
            pad_num = w_pad_num + h_pad_num
        else:
            if x.size(-2) % 2 == 0:
                pad_num = self.even_pad_num[0]
            else:
                pad_num = self.pad_num[0]

        return F.pad(x, pad_num)

def get_same_padding(kernel_size, stride, dilation):
    # todo: support 2d conv
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2

def get_tf_pad(kernel_size, stride):
    pad_config = []
    even_pad_config = []
    for i in range(len(kernel_size)):
        assert kernel_size[i] % 2 == 1
        pad_num_i = kernel_size[i] // 2
        pad_config.append([pad_num_i, pad_num_i])
        if stride[i] == 2:
            even_pad_config.append([pad_num_i - 1, pad_num_i])
        else:
            assert stride[i] == 1
            even_pad_config.append([pad_num_i, pad_num_i])
    return pad_config, even_pad_config

def create_pad_mask(lens, max_len=None):
    mask = torch.arange(max_len).to(lens.device) >= lens.unsqueeze(-1)
    return mask

def identity(x):
    return x  