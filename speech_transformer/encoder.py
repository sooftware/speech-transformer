# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Any

from speech_transformer.attention import MultiHeadAttention
from speech_transformer.convolution import VGGExtractor
from speech_transformer.embeddings import PositionalEncoding
from speech_transformer.mask import get_attn_pad_mask
from speech_transformer.modules import Linear, LayerNorm
from speech_transformer.sublayers import AddNorm, PositionWiseFeedForwardNet


class SpeechTransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            ffnet_style: str = 'ff'         # style of feed forward network
    ) -> None:
        super(SpeechTransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, self_attn_mask: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        output, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output = self.feed_forward(output)
        return output, attn


class SpeechTransformerEncoder(nn.Module):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        d_model: dimension of model (default: 512)
        input_dim: dimension of feature vector (default: 80)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoder layers (default: 6)
        num_heads: number of attention heads (default: 8)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
        dropout_p:  probability of dropout (default: 0.3)
        pad_id: identification of pad token (default: 0)

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths
    """

    def __init__(
            self,
            d_model: int = 512,
            input_dim: int = 80,
            d_ff: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            ffnet_style: str = 'ff',
            dropout_p: float = 0.3,
            pad_id: int = 0,
    ) -> None:
        super(SpeechTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.conv = VGGExtractor(input_dim)
        self.input_proj = Linear(self.conv.get_output_dim(), d_model)
        self.input_layer_norm = LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [SpeechTransformerEncoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)]
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor = None) -> Tuple[Tensor, Tensor]:
        conv_outputs, output_lengths = self.conv(inputs, input_lengths)

        self_attn_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))

        outputs = self.input_layer_norm(self.input_proj(conv_outputs))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        return outputs, output_lengths
