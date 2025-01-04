###############################################################################
# File: dataset.py
#
# Description:
#  This module contains a custom PyTorch Dataset class (BERTDataset) for
#  multi-label emotion classification. It leverages HuggingFace tokenizer
#  for text tokenization. Batches from this dataset align well with GPU
#  usage, especially if batch_size is a multiple of 32 (warp size).
###############################################################################

import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    """
    A PyTorch Dataset that transforms text samples into token IDs, 
    attention masks, and multi-label targets.
    
    Args:
      df (pd.DataFrame): DataFrame containing at least 'Text' column
        and label columns.
      tokenizer (transformers.Tokenizer): HuggingFace tokenizer for
        Roberta or any BERT-like model.
      max_len (int): Maximum token length for truncation/padding.
      target_cols (list): Column names for emotion labels.
    """
    def __init__(self, df, tokenizer, max_len, target_cols):
        self.df = df.reset_index(drop=True)
        self.texts = self.df.Text.astype(str)
        self.targets = self.df[target_cols].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Returns:
          dict with 'ids', 'mask', and 'targets' tensors.
        """
        text = self.texts[idx].strip()
        if not text:
            text = " "

        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }
