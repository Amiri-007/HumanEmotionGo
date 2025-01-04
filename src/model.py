###############################################################################
# File: model.py
# Author: Michael R. Amiri
# Date: 2025-01-04
#
# Description:
#  Defines a CustomRobertaModel with:
#   - RoBERTa base (roberta-large)
#   - MLP block
#   - Self-attention (token-level weighting)
#   - Multi-layer classification head
#
#  The forward pass returns logits and various intermediate representations.
#  Designed to leverage CUDA and utilize PyTorch's standard kernels, 
#  ensuring good warp occupancy if the user sets batch size as a multiple of 32.
###############################################################################

import torch
import torch.nn as nn
from transformers import AutoModel

class CustomRobertaModel(nn.Module):
    """
    A custom architecture built on RoBERTa-large, extended with:
      - An MLP block
      - A self-attention weighting layer
      - A multi-layer classification head
    
    Returns:
      logits, all_hidden_states, sequence_output, weighted_out, penultimate_feats
    """
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        # roberta-large with last hidden states accessible
        self.roberta = AutoModel.from_pretrained('roberta-large', output_hidden_states=True)
        hidden_size = 1024  # dimension for roberta-large

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Self-attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate/2),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Manually initialize linear/LayerNorm layers in MLP, attention, classifier 
        for more controlled training behavior. Uses Xavier for Linear and 
        ones/zeros for LayerNorm.
        """
        for module in [self.mlp, self.attention, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, ids, mask):
        """
        Forward pass, leveraging last 4 hidden layers of RoBERTa.
        
        Args:
          ids (torch.LongTensor): Token IDs of shape (batch_size, max_len).
          mask (torch.LongTensor): Attention mask of shape (batch_size, max_len).
        
        Returns:
          logits (torch.FloatTensor): Shape (batch_size, num_classes).
          all_hidden_states (tuple): All hidden states from roberta.
          sequence_output (torch.FloatTensor): The MLP-transformed last-4-layers-avg.
          weighted_out (torch.FloatTensor): Token-wise self-attention weighted sum.
          penultimate_feats (torch.FloatTensor): Features right before final linear.
        """
        outputs = self.roberta(ids, attention_mask=mask, return_dict=True)
        all_hidden_states = outputs.hidden_states

        # Average the last 4 layers
        last_four = torch.stack(all_hidden_states[-4:])
        concat_features = torch.mean(last_four, dim=0)  # [batch, seq_len, hidden]

        # MLP
        sequence_output = self.mlp(concat_features)

        # Self-attention
        attention_weights = self.attention(sequence_output)  # [batch, seq_len, 1]
        weighted_out = torch.sum(attention_weights * sequence_output, dim=1)  # [batch, hidden]

        # Step-by-step classification
        x = self.classifier[0](weighted_out) # 1024 -> 512
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)           # 512 -> 256
        x = self.classifier[5](x)
        x = self.classifier[6](x)
        x = self.classifier[7](x)
        penultimate_feats = x.clone()       # keep a copy for analysis

        logits = self.classifier[8](x)      # final linear

        return logits, all_hidden_states, sequence_output, weighted_out, penultimate_feats
