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
from typing import Any, Optional
from speech_transformer.modules import (
    Linear,
    LayerNorm,
    MaskConv2d,
)


class AddNorm(nn.Module):
    """
    Add & Normalization layer proposed in "Attention Is All You Need".
    Transformer employ a residual connection around each of the two sub-layers,
    (Multi-Head Attention & Feed-Forward) followed by layer normalization.
    """
    def __init__(self, sublayer: nn.Module, d_model: int = 512) -> None:
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)

        if isinstance(output, tuple):
            return self.layer_norm(output[0] + residual), output[1]

        return self.layer_norm(output + residual)


class PositionWiseFeedForwardNet(nn.Module):
    """
    Position-wise Feedforward Networks proposed in "Attention Is All You Need".
    Fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    Another way of describing this is as two convolutions with kernel size 1.
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048,
                 dropout_p: float = 0.3, ffnet_style: str = 'ff') -> None:
        super(PositionWiseFeedForwardNet, self).__init__()
        self.ffnet_style = ffnet_style.lower()
        if self.ffnet_style == 'ff':
            self.feed_forward = nn.Sequential(
                Linear(d_model, d_ff),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                Linear(d_ff, d_model),
                nn.Dropout(dropout_p),
            )

        elif self.ffnet_style == 'conv':
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        else:
            raise ValueError("Unsupported mode: {0}".format(self.mode))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.ffnet_style == 'conv':
            output = self.conv1(inputs.transpose(1, 2))
            output = self.relu(output)
            return self.conv2(output).transpose(1, 2)

        return self.feed_forward(inputs)


class CNNExtractor(nn.Module):
    """
    Provides inteface of convolutional extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    supported_activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU()
    }

    def __init__(self, activation: str = 'hardtanh') -> None:
        super(CNNExtractor, self).__init__()
        self.activation = CNNExtractor.supported_activations[activation]

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DeepSpeech2Extractor(CNNExtractor):
    """
    DeepSpeech2 extractor for automatic speech recognition described in
    "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin" paper
    - https://arxiv.org/abs/1512.02595
    """
    def __init__(self, activation: str = 'hardtanh', mask_conv: bool = False) -> None:
        super(DeepSpeech2Extractor, self).__init__(activation)
        self.mask_conv = mask_conv
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            self.activation
        )
        if mask_conv:
            self.conv = MaskConv2d(self.conv)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Optional[Any]:
        if self.mask_conv:
            return self.conv(inputs, input_lengths)
        return self.conv(inputs)


class VGGExtractor(CNNExtractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """
    def __init__(self, activation: str = 'hardtanh', mask_conv: bool = False):
        super(VGGExtractor, self).__init__(activation)
        self.mask_conv = mask_conv
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            self.activation,
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            self.activation,
            nn.MaxPool2d(2, stride=2)
        )
        if mask_conv:
            self.conv = MaskConv2d(self.conv)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Optional[Any]:
        if self.mask_conv:
            return self.conv(inputs, input_lengths)
        return self.conv(inputs)
