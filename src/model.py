# src/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class CustomRobertaLarge(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-large", output_hidden_states=True)
        hidden_size = 1024  # roberta-large hidden dimension

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

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
        all_hidden_states = outputs.hidden_states  # tuple of length 25 for roberta-large

        # Average of last 4 layers
        last_four = torch.stack(all_hidden_states[-4:])  # shape [4, batch, seq_len, hidden]
        concat_features = torch.mean(last_four, dim=0)   # [batch, seq_len, hidden]

        # MLP block
        sequence_output = self.mlp(concat_features)

        # attention pooling
        att_w = self.attention(sequence_output)   # [batch, seq_len, 1]
        weighted_out = torch.sum(att_w * sequence_output, dim=1)  # [batch, hidden]

        # Step through classifier to obtain penultimate
        x = self.classifier[0](weighted_out) 
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        x = self.classifier[6](x)
        x = self.classifier[7](x)
        penultimate_feats = x.clone()

        logits = self.classifier[8](x)
        return logits, all_hidden_states, sequence_output, weighted_out, penultimate_feats
