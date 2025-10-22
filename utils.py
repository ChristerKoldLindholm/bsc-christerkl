import numpy as np
from pathlib import Path
import torch 
import torchaudio as ta
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloader built based on PyTorch tutorial.
# Uses helper max_len_collate() function below.
class AudioDataset(Dataset):
    """Audio dataset class suited for the hydroacoustic dataset from 
    Inglefield Bredning Fjord, Greenland. Helper function 
    max_len_collate() determines the final output batch contents: 
    (waves, srs, paths, lengths).
    Returns:
        A dictionary with keys:
            "waveform": torch.Tensor of shape [C, T]
            "sample_rate": int
            "path": str, path to the audio file
    """
    def __init__(self, root_dir, target_sr=64000, skip_secs=5, mode="crop", max_secs=None):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.rglob("*.wav")) # Root data folder.
        self.target_sr = target_sr # Can change from original raw 64 kHz to common 16 kHz.
        self.skip_secs = skip_secs # Skip a recording's first corrupted seconds.
        self.mode = mode # "crop" or "mute" the corrupted recording segment.
        self.max_frames = int(target_sr * max_secs) if max_secs else None # Truncation for plotting.

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        wf, sr = ta.load(path)
        # For the entire dataset, each waveform shape is [C, T] where C = 1 (mono), 
        # T (the number of frames) is approximately = 172 million,
        # and is recorded at sample rate = 64 kHz.
        
        # Skip corrupted beginning of every recording.
        s_idx = int(self.skip_secs * sr)
        if self.mode == "crop":
            wf = wf[:, s_idx:] if wf.shape[1] > s_idx else torch.zeros((wf.shape[0], 1))
        elif self.mode == "mute":
             wf = wf.clone()
             s = min(s_idx, wf.shape[-1])
             wf[:, :s] = 0.0

        if sr != self.target_sr:
            wf = ta.functional.resample(wf, sr, self.target_sr)
            sr = self.target_sr

        if self.max_frames is not None:
            max_len = int(self.max_secs * sr)
            wf = wf[:, :max_len]

        item = {"waveform": wf, "sample_rate": sr, "path": str(path)}

        return item

# Collates tensors in a batch by padding to the max length tensor.
def max_len_collate(batch):
    # waves, srs, paths = zip(*batch)
    waves = [b["waveform"] for b in batch] # List of [C, T] tensors.
    C = waves[0].shape[0]
    max_len = max([w.shape[-1] for w in waves])
    padded = []
    lengths = []
    for w in waves:
        pad_T = max_len - w.shape[-1]
        padded.append(F.pad(w, (0, pad_T))) # Pad last dimension.
        lengths.append(w.shape[-1])
    waves = torch.stack(padded) # [B, C, T]

    output = {
        "waveforms": waves,
        "sample_rates": batch[0]["sample_rate"], # All sample rates are the same.
        "paths": [b["path"] for b in batch],
        "lengths": torch.tensor(lengths), # Keep original lengths for later data processing.
    }
    return output 

# A preprocessing pipeline class for audio features. 
class PipelineSpecgram(torch.nn.Module):
    def __init__(self, specgram_config:dict):
        super().__init__()
        # Basic spectrogram settings.
        self.sample_rate = specgram_config["sample_rate"]
        self.n_fft = specgram_config["n_fft"]
        self.win_length = specgram_config["win_length"]
        self.hop_length = specgram_config["hop_length"]
        self.window_fn = specgram_config["window_fn"]
        # Optional spectrogram settings.
        self.resample_rate = specgram_config.get("resample_rate", None)
        self.mel_bins = specgram_config.get("n_mels", None)
        self.power = specgram_config.get("power", 2.0)
        self.to_db = T.AmplitudeToDB(stype="power") # Decibel conversion.

        if self.resample_rate is not None and self.resample_rate != self.sample_rate:
            self.resample = T.Resample(orig_freq=self.sample_rate, new_freq=self.resample_rate)
            self.effective_sr = self.resample_rate
        else:
            self.resample = torch.nn.Identity() # Placeholder identity.
            self.effective_sr = self.sample_rate

        # Setup spectrogram. 
        self.spec = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=self.window_fn,
            power=self.power,
            center=True,
            pad_mode="reflect"
        )

        # Mel scale transformation.
        if self.mel_bins is not None:
            # Mel scale produces power mel bands.
            self.mel_scale = T.MelScale(
                n_mels = self.mel_bins,
                sample_rate = self.effective_sr,
                n_stft = self.n_fft // 2 + 1, # = n_freqs
                f_min = 0.0,
                f_max = self.effective_sr / 2.0, # = 32 kHz according to the Nyquist theorem.
                mel_scale="htk", # Default is "htk". 
                # norm="slaney"
            )
        else:
            self.mel_scale = None 
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.resample(waveform) # Skips resampling if Identity was called.
        spec = self.spec(x) # Applies the power spectrogram settings to each input waveform.
        # Returns shape [C, F, T] for channels, freq bins and time frames.
        
        # p_ref is in Pa^2. Divide by micro-Pascals squared to get reference level.
        spec_ref = spec / (1e-6**2) 

        if self.mel_scale is not None: 
            mel = self.mel_scale(spec_ref)
            # Returns shape [C, M, T] for channels, mel bins and time frames. 
            mel_db = self.to_db(mel) # AmplitudeToDB(stype='power') produces log-mel dB. 
            return mel_db
        else: 
            spec_db = self.to_db(spec_ref)
            return spec_db

# Helper function to reduce the number of sample points in audio data tensors.
def reduce_tensor(w, max_pts):
    """
    Reduces a 1D tensor w to exactly max_pts elements. 
    If w has more than max_pts elements, it is downsampled using fixed indices.
    If w has fewer than max_pts elements, it is padded with zeros at the end.
    Args:
        w: 1D torch.Tensor.
        max_pts: Maximum number of elements in the output tensor.
    Returns:
        w_small: Reduced or padded 1D tensor of length max_pts.
    """
    # Downsample to max_pts using fixed indices.
    n_elms = w.numel()
    if n_elms == max_pts:
        return w 
    
    if n_elms == 0:
        return torch.zeros(max_pts, device=w.device, dtype=w.dtype)
    
    if n_elms >= max_pts:
        k = torch.arange(max_pts, device=w.device)
        idx = torch.floor(k.to(torch.float32) * (n_elms / float(max_pts))).to(torch.long)
        # idx = torch.linspace(0, n_elms - 1, steps=max_pts, device=w.device).to(torch.long)
        w_small = w.index_select(0, idx)
    else:
        # Padding needed at dataset level to build X (data loader pads at batch-level).
        w_small = F.pad(w, (0, max_pts - n_elms))
    return w_small

# Builds feature matrix Z as inputs to clustering models.
def tensors_to_array(dataloader, transform, max_pts=None, dtype=np.float32, device=device):
    """
    Builds a (N, max_pts) feature matrix Z of type NumPy array by reducing PyTorch 
    waveform tensors. Z used as input for clustering algorithms. The size of Z can 
    be reduced by downsampling based on a maximum number of points per audio recording.
    Args:
        dataloader: PyTorch DataLoader batches of audio waveforms.
        max_pts: Maximum number of points per feature.  
    Returns:
        X: NumPy array of shape (n samples, n features).
        ids: List of audio file identifiers corresponding to each row in X.
    """

    # if device is not None: 
    #     device = next(transform.parameters()).device if hasattr(transform, "parameters") else torch.device("cpu")

    rows, ids = [], []
    transform.eval()

    with torch.no_grad(): # Disables gradient computations
        for batch in dataloader:
            waves = batch["waveforms"]
            paths = batch["paths"]
            lengths = batch["lengths"]

            B, C, Tn = waves.shape

            for b in range(B):
                L = int(lengths[b].item()) if lengths is not None else Tn
                L = max(0, min(L, Tn))
                w = waves[b, :, :L][0]

                # n_elms = w.numel()
                if max_pts is not None:
                    T0 = w.numel()
                    if T0 == 0:
                        w = torch.zeros(max_pts, dtype=w.dtype)
                    elif T0 > max_pts:
                        k = torch.arange(max_pts, device=w.device)
                        idx = torch.floor(k.to(torch.float32) * (T0 / float(max_pts))).to(torch.long)
                        # idx = torch.linspace(0, w.numel()-1, steps=max_pts).long()
                        w = w.index_select(0, idx)
                    elif T0 < max_pts:
                        w = F.pad(w, (0, max_pts - T0)) 
                        # pad = max_pts - w.numel()
                        # w = F.pad(w, (0, pad))

                x = w.view(1, 1, -1).to(device=device, dtype=torch.float32)
                
                feat = transform(x).squeeze(0)  # [C, F, T]

                mu = feat.mean(dim=-1, keepdim=False)
                sig = feat.std(dim=-1, keepdim=False)
                vec = torch.cat([mu, sig], dim=0)
                vec = vec.reshape(-1)

                rows.append(vec.detach().cpu().numpy().astype(dtype))

                p = paths[b]
                ids.append(Path(p).name if isinstance(p, (str, Path)) else str(p))

    # Concatenate rows into Z.
    # Z: (n samples, n features.)
    Z = np.stack(rows, axis=0)
    return Z, ids

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