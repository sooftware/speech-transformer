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
from typing import Optional, Union

from speech_transformer.beam_decoder import BeamTransformerDecoder
from speech_transformer.decoder import SpeechTransformerDecoder
from speech_transformer.encoder import SpeechTransformerEncoder


class SpeechTransformer(nn.Module):
    """
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        num_classes (int): the number of classfication
        d_model (int): dimension of model (default: 512)
        input_dim (int): dimension of input
        pad_id (int): identification of <PAD_token>
        eos_id (int): identification of <EOS_token>
        d_ff (int): dimension of feed forward network (default: 2048)
        num_encoder_layers (int): number of encoder layers (default: 6)
        num_decoder_layers (int): number of decoder layers (default: 6)
        num_heads (int): number of attention heads (default: 8)
        dropout_p (float): dropout probability (default: 0.3)
        ffnet_style (str): if poswise_ffnet is 'ff', position-wise feed forware network to be a feed forward,
            otherwise, position-wise feed forward network to be a convolution layer. (default: ff)

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.

    Returns: output
        - **output**: tensor containing the outputs
    """

    def __init__(
            self,
            num_classes: int,
            d_model: int = 512,
            input_dim: int = 80,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            d_ff: int = 2048,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dropout_p: float = 0.3,
            ffnet_style: str = 'ff',
            extractor: str = 'vgg',
            joint_ctc_attention: bool = False,
            max_length: int = 128,
    ) -> None:
        super(SpeechTransformer, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_classes = num_classes
        self.extractor = extractor
        self.joint_ctc_attention = joint_ctc_attention
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

        self.encoder = SpeechTransformerEncoder(
            d_model=d_model,
            input_dim=input_dim,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id,
        )

        self.decoder = SpeechTransformerDecoder(
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
        )

    def set_beam_decoder(self, batch_size: int = None, beam_size: int = 3):
        """ Setting beam search decoder """
        self.decoder = BeamTransformerDecoder(
            decoder=self.decoder,
            batch_size=batch_size,
            beam_size=beam_size,
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Optional[Tensor] = None,
            target_lengths: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        logits = None
        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        if isinstance(self.decoder, BeamTransformerDecoder):
            predictions = self.decoder(encoder_outputs, encoder_output_lengths)
        else:
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                targets=targets,
                teacher_forcing_ratio=0.0,
                target_lengths=target_lengths,
            )
            predictions = logits.max(-1)[1]

        return predictions, logits
