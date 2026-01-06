import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path 
import torch
import torchaudio as ta

# Plots the pure waveform vector.
def plot_waveform(waveform, sample_rate):
    """Plots the raw waveform (mono channel)."""
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
def plot_waveform_fast(waveform, sample_rate, skip_secs:int=0, max_pts:int=30_000, title:str="Waveform", ax=None):
    """
    Plots the raw waveform (mono channel) with truncation for faster visualization.
    """

    n_channels, n_frames = waveform.shape
    # Truncation steps only for visualizing waveforms.
    stride = max(1, n_frames // max_pts)
    s_idx = int(skip_secs * sample_rate)
    waveform = waveform[:, s_idx:] if n_frames > s_idx else torch.zeros((n_channels, 1))
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

def plot_specgram(waveform, sample_rate, config:dict, title:str="Spectrogram", ax=None):
    """
    Plots the spectrogram of a waveform (mono channel).
    """
    
    n_channels, n_frames = waveform.shape
    wf = waveform.numpy()

    if ax is None: 
        fig, ax = plt.subplots(n_channels, 1, figsize=(7,6))
    else: 
        fig = None


    ax.specgram(wf[0], Fs=sample_rate
                , NFFT=config['n_fft']
                , noverlap=config['n_fft'] - config['hop_length']
                , window=config['window_fn'](config['win_length']).numpy()
                , mode='magnitude'
                , scale_by_freq=True
                , sides='default'
                , cmap='viridis')

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (kHz)")
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{y/1000:.0f}k" for y in yticks])
    ax.set_title(title)
    plt.tight_layout()

    return fig, ax

def plot_nm_waveforms(n_rows:int, m_cols:int, wavelist:list, fname:str="waveform_graph", fig_size=(9, 9), data_path=Path, graph_path=Path):
    """Plots n x m waveforms in a combined figure."""
    fig, axes = plt.subplots(n_rows, m_cols, figsize=fig_size)
    axes = axes.flatten()

    max_points = 30_000
    for i, wav in enumerate(wavelist):
        if i >= len(axes):
            break 

        p = Path(data_path / wav)
        wf, sr = ta.load(str(p))
        plot_waveform_fast(wf, sr, max_pts=max_points, title=p.name, ax=axes[i])

    plt.savefig(fname=graph_path / f"{fname}.png", bbox_inches='tight', dpi=300)
    plt.tight_layout()
    
    return fig, axes

def plot_nm_specgrams(n_rows:int, m_cols:int, wavelist:list, fname:str="spectro_graph", fig_size=(9, 9), data_path=Path, graph_path=Path):
    """Plots n x m spectrograms in a combined figure."""
    fig, axes = plt.subplots(n_rows, m_cols, figsize=fig_size)
    axes = axes.flatten()

    for i, wav in enumerate(wavelist):
        if i >= len(axes):
            break

        p = Path(data_path / wav)
        wf, sr = ta.load(str(p))
        plot_specgram(wf, sr, title=p.name, ax=axes[i])

    if graph_path is None:
        graph_path = Path().resolve().parent / "graphs"
    plt.savefig(fname=graph_path / f"{fname}.png", bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.tight_layout()

    return fig, axes

def plot_features(feats: torch.Tensor, transf: torch.nn.Module, specgram_config: dict
                  , path: str=None, title="Log-mel spectrogram", ax=None, unit="freqs"
                  , duration_sec: float=None, t_frames_original: int=None, mel_idx_start:int=0
                  , n_mels_full: int=None):
    """
    Plots a spectrogram from transformed acoustic features.
    feats = Tensor of shape [C, T, F] where C is number of channels (always 1)
    , F is number of frequency (mel) bins and T is number of time frames.
    transf = A transformation class or module.
    """
    
    if feats.ndim == 2:
        Z = feats 
    elif feats.ndim == 3:
        Z = feats[0] # Select the first channel.
    elif feats.ndim == 4:
        Z = feats[0, 0]
    else:
        raise ValueError("Feature tensor dimensions are supported up to 4D.")
    
    # To numpy for plotting.
    Z_np = Z.detach().float().cpu().numpy() # [T, F]    
    T, F = Z_np.shape

    sr = getattr(transf, "effective_sr", specgram_config.get("sample_rate"))
    hop = getattr(transf, "hop_length", specgram_config.get("hop_length"))
    f_min    = getattr(transf, "f_min", specgram_config.get("f_min", 0.0))
    f_max    = getattr(transf, "f_max", specgram_config.get("f_max", sr / 2.0))

    if sr is None or hop is None:
        raise ValueError("Sample rate and hop length must be defined in either transf or specgram_config.")
    if duration_sec is not None:
        t_max = float(duration_sec)
    elif t_frames_original is not None:
        t_max = (t_frames_original * hop) / sr 
    else:
        t_max = (T * hop) / sr     

    Z_plot = Z_np.T # [F, T]
    bounds = [0.0, t_max, 0, F]

    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None 

    im = ax.imshow(Z_plot, origin="lower", aspect="auto", extent=bounds)
    ax.set_xlabel("Time (s)")

    # y-axis can be either labels as mel bins or frequencies.
    is_mel = getattr(transf, "mel_scale", None) is not None
    yticks = np.linspace(0, F -1, min(F, 6)).astype(int)
    ax.set_yticks(yticks)

    if is_mel:
        yticks = np.linspace(0, F-1, 6).astype(int)
        ax.set_yticks(yticks)

        # If mel bins.
        if unit == "bins": 
            ax.set_ylabel("Mel bin")
            ytick_labels = [str(mel_idx_start + i) for i in yticks]
            ax.set_yticklabels(ytick_labels)
            
        # If frequencies.
        elif unit == "freq": 
            ax.set_ylabel("Frequency (kHz)") 
            f_min = specgram_config.get("f_min", 0.0)
            f_max = specgram_config.get("f_max", sr / 2.0)

            if n_mels_full is not None and (mel_idx_start > 0 or n_mels_full != F):
                fb_full = ta.functional.melscale_fbanks(
                    n_freqs=specgram_config["n_fft"]//2 + 1, f_min=f_min, f_max=f_max, n_mels=n_mels_full
                    , sample_rate=sr, mel_scale="htk"
                    )
                peak_full = fb_full.argmax(dim=0)
                peak_slice = peak_full[mel_idx_start: mel_idx_start + F]
                center_freqs = (peak_slice.float() * (sr / specgram_config["n_fft"])).cpu().numpy()
            else:
                # Compute the mel filter bank center frequencies and use as y-axis labels.
                fb = ta.functional.melscale_fbanks(
                    n_freqs=specgram_config["n_fft"]//2 + 1, f_min=f_min, f_max=f_max, n_mels=n_mels_full
                    , sample_rate=sr, mel_scale="htk"
                    )
                peak_bins = fb.argmax(dim=0)
                center_freqs = (peak_bins.float() * (sr / specgram_config["n_fft"])).cpu().numpy()

            ytick_labels = np.round(center_freqs[yticks] / 1000.0, 1)  # in kHz.
            ax.set_yticklabels(ytick_labels)

        # Color bar. 
        cbar_label = "dB re 1 µPa" if hasattr(transf, "to_db") else "Amplitude"
    # If linear spectrogram.
    else:
        freqs = np.linspace(0, sr / 2, F)
        ax.set_yticklabels((freqs[yticks] / 1000.0).round(1))
        ax.set_ylabel("Frequency (kHz)")
        cbar_label = "dB" if hasattr(transf, "to_db") else "Power"
    
    cbar = plt.colorbar(im, ax=ax, format="%+2.0f")
    cbar.set_label(cbar_label)
    ax.set_title(title)
    plt.tight_layout()

    if path is not None:
        fig_to_save = fig if fig is not None else ax.figure
        fig_to_save.savefig(path, dpi=300)

    return (fig if fig is not None else ax.figure), ax

