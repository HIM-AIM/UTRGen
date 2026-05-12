# regression_dataset.py
from torch.utils.data import Dataset
import torch

class mRNA_RegressionDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        """
        :param tokenized_data: got "input_ids" and "attention_mask"
        :param labels: sequences labels
        """
        assert len(tokenized_data) == len(labels), "Data and label quantity do not match"
        self.data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"] 
        attention_mask = item["attention_mask"]
        
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long), 
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": label_tensor
        } 

if __name__ == "__main__":
    import numpy as np
    data = [{"input_ids": [1,2,3,4], "attention_mask": [1,1,1,1]},
            {"input_ids": [5,6,7,8], "attention_mask": [1,1,1,1]}]
    labels = np.array([[0.5, 0.6], [ 0.8, float('nan')]])
    dataset = mRNA_RegressionDataset(data, labels)
    for i in range(len(dataset)):
        print(dataset[i])