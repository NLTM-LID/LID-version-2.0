import random
import torch


class RandomResizedCrop(torch.nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, audio):
        max_samples = audio.shape[-1]
        shape_0 = audio.shape[0]
        start_idx = random.randint(0, max_samples - self.n_samples)
        audio = audio[..., start_idx : start_idx + self.n_samples]
        audio_l = torch.cat([torch.zeros((shape_0, start_idx)), audio], dim=-1)
        audio_lr = torch.cat([audio_l, torch.zeros((shape_0, max_samples - audio_l.shape[-1]))], dim=-1)
        return audio_lr
