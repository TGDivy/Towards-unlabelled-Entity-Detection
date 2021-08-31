# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """
import sys
import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.nn import TripletMarginLoss
from torch.nn import CosineSimilarity
from torch.nn import BatchNorm1d
from torch.nn import PairwiseDistance
import torch.nn.functional as F
from torch.nn import Sigmoid

from transformers.modeling_utils import (PreTrainedModel)
from transformers import RobertaConfig, RobertaModel

class Roberta(RobertaModel):
    def __init__(self, config):
        super(Roberta, self).__init__(config)

        self.sequence_length = 192#config.max_sequence_length
        self.hidden_size = 768

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.binary_cross_entropy = F.binary_cross_entropy
        self.cross_entropy = CrossEntropyLoss()
        self.mse = MSELoss()
        
        self.weights = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def compute_vector(self, sequence, batch_size, normalize=False):
        logits = 0
        total_weight = 0
        for i, weight in enumerate(self.all_weights):
            weight_temp = weight(sequence).view(batch_size, self.sequence_length)
            total_weight+= weight_temp
        logits = self.sigmoid(total_weight).view(batch_size, 1, self.sequence_length) #* self.weighted_weight[i]
        v_ = torch.matmul(logits,  sequence).view(batch_size, self.hidden_size)
        if normalize:
            v_ = v_/torch.norm(v_, dim=1, keepdim=True)
        return v_, logits

    def sequence_tagging(self, sequence, weight, labels, batch_size, threshold=None):
        label = torch.clamp(labels.view(batch_size, self.sequence_length), min=0, max=3)
        sentence_weight = weight(sequence).view(batch_size, self.sequence_length)
        sentence_weight_logits = self.sigmoid(sentence_weight)*3

        loss = self.mse(sentence_weight_logits, label)
        
        return loss, sentence_weight_logits

    def forward(self, batch, train=True):
        r"""
            Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using :class:`transformers.RobertaTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.__call__` for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                `What are attention masks? <../glossary.html#attention-mask>`__
            token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
                Segment token indices to indicate first and second portions of the inputs.
                Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
                corresponds to a `sentence B` token
                `What are token type IDs? <../glossary.html#token-type-ids>`_
            position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, config.max_position_embeddings - 1]``.
                `What are position IDs? <../glossary.html#position-ids>`_
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules.
                Mask values selected in ``[0, 1]``:
                :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
                If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        """
        total_loss = 0
        batch_size = 16

        batch_size = batch["inputs"][0].size()[0]
        
        outputs = self.roberta(batch["inputs"][0], attention_mask=batch["inputs"][1])

        sequence = outputs[0]
        label = batch["labels"]
        loss, logits = self.sequence_tagging(sequence, self.weights, label, batch_size)
        total_loss += loss

        outputs = (total_loss, logits)
        
        return outputs