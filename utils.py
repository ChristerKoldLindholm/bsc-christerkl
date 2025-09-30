import torch 
import torchaudio as ta 
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from pathlib import Path

# Dataloader built based on PyTorch tutorial.
class AudioDataset(Dataset):
    def __init__(self, root, target_sr=64000, max_secs=None):
        self.files = list(Path(root).rglob("*.wav")) # Root data folder.
        self.target_sr = target_sr # Can change from original raw 64 kHz to common 16 kHz.
        self.max_frames = int(target_sr * max_secs) if max_secs else None # Truncation for plotting.

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        wf, sr = ta.load(path)

        if sr != self.target_sr:
            wf = ta.functional.resample(wf, sr, self.target_sr)
            sr = self.target_sr 
        
        if self.max_frames: 
            wf = wf[:, :self.max_frames]

        return wf, sr, str(path)
    
def max_len_collate(batch):
    waves, srs, paths = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in waves])
    max_len = int(lengths.max())
    pad_waves = [F.pad(w, (0, max_len - w.shape[-1])) for w in waves]
    # Keep original lengths for later data processing. 
    items = [(pad_waves[i], srs[i], paths[i], lengths[i]) for i in range(len(batch))]
    
    return default_collate(items)