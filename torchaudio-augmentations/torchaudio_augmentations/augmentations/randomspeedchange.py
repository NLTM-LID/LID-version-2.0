import random
import torchaudio
import torch

class RandomSpeedChange(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0: # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio
 
