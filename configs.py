import torch

# Configurations for mel-spectrogram feature generation shared
# across the project.
def get_specgram_config():

    configs = dict(
        sample_rate=64_000,
        n_fft=1024, # Changed from 512.
        win_length=1024, # From 512.
        hop_length=512, # From 256
        window_fn=torch.hann_window,
        resample_rate=None,
        n_mels=128, # From 256.
        power=2.0, 
        center=True, # Default.
        pad_mode="reflect", # Default.
        f_min=0.0,
        f_max=None
    )
    configs["f_max"] = configs['sample_rate'] / 2.0

    return configs