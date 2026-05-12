from torch.utils.data import Dataset
import torch
import re
from Bio import SeqIO


class mRNA_Dataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=256):
        self.max_length = max_length
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.data = []
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            print(f"Loading sequences from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for record in SeqIO.parse(f, 'fasta'):
                    seq = str(record.seq)
                    if seq:
                        self.data.append(seq)
        print(f'total loading sequences is:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index]
        if seq:
            encoded = self.tokenizer(seq,
                                     max_length=255,
                                     padding=False,
                                     truncation=True,
                                     add_special_tokens=False)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            # Add EOS token
            input_ids.append(self.tokenizer.sep_token_id)
            attention_mask.append(1)

            # Pad to max_length
            current_length = len(input_ids)
            if current_length < self.max_length:
                pad_length = self.max_length - current_length
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            else:
                input_ids = input_ids[:self.max_length - 1] + [self.tokenizer.sep_token_id]
                attention_mask = attention_mask[:self.max_length - 1] + [1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


class mRNA_MFE_Dataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.data = []
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            print(f"Loading sequences from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for record in SeqIO.parse(f, 'fasta'):
                    seq = str(record.seq)
                    mfe_norm = self.extract_mfe_norm(record.description)
                    if seq:
                        self.data.append((seq, mfe_norm))

    def extract_mfe_norm(self, header):
        m = re.search(r"Norm_MFE:\s*([-\d\.]+)", header)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return 0.0
        return 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, mfe_norm = self.data[index]
        encoded = self.tokenizer.encode(seq)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        mfe_norm_tensor = torch.tensor(mfe_norm, dtype=torch.float)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "mfe_norm": mfe_norm_tensor
        }
