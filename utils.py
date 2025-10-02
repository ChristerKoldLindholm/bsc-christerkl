import matplotlib.pyplot as plt
from pathlib import Path
import torch 
import torchaudio as ta 
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate

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

# Collates tensors in a batch by padding to the max length tensor.
def max_len_collate(batch):
    waves, srs, paths = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in waves])
    max_len = int(lengths.max())
    pad_waves = [F.pad(w, (0, max_len - w.shape[-1])) for w in waves]
    # Keep original lengths for later data processing. 
    items = [(pad_waves[i], srs[i], paths[i], lengths[i]) for i in range(len(batch))]
    
    return default_collate(items)

# Plots the pure waveform vector.
def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_ax = torch.arange(0, num_frames) / sample_rate

    fig, ax = plt.subplots(num_channels, 1)
    ax.plot(time_ax, waveform[0], linewidth=1, color="royalblue", alpha=0.8)
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    fig.suptitle("Waveform")

# Plots a waveform vector points at defined stride intervals for faster visualization.
def plot_waveform_fast(waveform, sample_rate, max_pts:int=30_000, title:str="Waveform", ax=None):
    n_channels, n_frames = waveform.shape
    # Truncation steps only for visualizing waveforms.
    stride = max(1, n_frames // max_pts) 
    wf_visual = waveform[:, ::stride]
    # Time in seconds: time = frames / sample_rate and strides are frames divided by steps.  
    t = torch.arange(wf_visual.shape[1], dtype=torch.float32) * (stride / sample_rate) 

    wf_visual = wf_visual.cpu().numpy()
    t = t.cpu().numpy()

    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 5))
    else: 
        fig = None

    ax.plot(t, wf_visual[0], linewidth=1, color="royalblue", alpha=0.8)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    plt.tight_layout()
    
    return fig, ax 

# Computes descriptive statistics: peak amplitudes, mean amplitudes, 
# root mean squares and zero-crossing rates
def compute_stats(w, sr, length, skip_secs):
            
            n_chans, n_frames = w.shape 

            empty_dict = dict(duration_sec = 0, peak=0.0, mean_abs=0.0, rms=0.0, zcr_hz=0.0)
            if n_frames <= 0:
                   return empty_dict

            s_idx = int(sr * float(skip_secs))
            s_idx = max(0, min(s_idx, n_frames)) # Ensure no 0 length files. 

            wf = w[:, :n_frames].clone()
            if s_idx > 0:
                wf[:, :s_idx] = 0.0 # Mute first corrupted seconds.
            
            # Duration in seconds.
            duration_sec = max((n_frames - s_idx) / float(sr), 0.0)

            # Wave without the muted clip.
            if s_idx < n_frames: 
                  wclip = wf[:, s_idx:] 
            else: 
                  return empty_dict

            if wclip.numel() == 0:
                  return empty_dict

            peak = wclip.abs().amax().amax().item()
            mean_abs = wclip.abs().mean().item()
            # Root mean square.
            rms = wclip.pow(2).mean().sqrt().item()
            # Zero-Crossing Rate.
            silence_band = 10e-11
            wclip_nz = torch.where(wclip == 0, silence_band, wclip)
            signs = torch.signbit(wclip_nz)

            if wclip.shape[1] >= 2:
                  changes = signs[:, 1:] ^ signs[:, :-1]
                  zc = changes.sum().item()
                  zcr_hz = (zc / (wclip_nz.shape[1] - 1)) * sr
            else:
                  zcr_hz = 0.0

            collected_stats = dict(
                  duration_sec = float(duration_sec),
                  peak_abs = float(peak),
                  mean_abs = float(mean_abs),
                  rms = float(rms),
                  zcr_hz = float(zcr_hz)
            )

            return collected_stats    