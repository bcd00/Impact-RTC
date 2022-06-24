import torch
from torch.utils.data import Dataset


class DensityDataset(Dataset):
    def __init__(self, df, cols, device):
        self.df = df
        self.cols = cols
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.df[self.cols].iloc[idx]).to(self.device).float(),
            'output': torch.tensor(self.df['size'].iloc[idx]).to(self.device).float().unsqueeze(0)
        }
