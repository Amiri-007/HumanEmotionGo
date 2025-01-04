# src/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class CustomRobertaModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        # roberta-large model with hidden_states
        self.roberta = AutoModel.from_pretrained('roberta-large', output_hidden_states=True)
        hidden_size = 1024  # roberta-large hidden dim

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Self-attention
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
            nn.Dropout(dropout_rate / 2),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize linear/LayerNorm in MLP, attention, classifier
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

        # step-by-step classifier
        x = self.classifier[0](weighted_out)   # Linear(1024->512)
        x = self.classifier[1](x)              # LayerNorm(512)
        x = self.classifier[2](x)              # Dropout
        x = self.classifier[3](x)              # GELU
        x = self.classifier[4](x)              # Linear(512->256)
        x = self.classifier[5](x)              # LayerNorm(256)
        x = self.classifier[6](x)              # Dropout
        x = self.classifier[7](x)              # GELU
        penultimate_feats = x.clone()          # [batch, 256]
        logits = self.classifier[8](x)         # final Linear(256->num_classes)

        return logits, all_hidden_states, sequence_output, weighted_out, penultimate_feats
