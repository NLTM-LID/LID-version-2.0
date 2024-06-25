#####################################################
################# Acknowledgements ##################
## This code is developed under the Natural Language Translation Mission (NLTM) consortium project "Speech Technologies in Indian Languages".
## The copyrights for this code belong to IIT Mandi and IIT Dharwad.
#####################################################
## @author : Sujeet Kumar
#####################################################

## loading important libraries
import sys
sys.path.append("../data2vec_aqc")
import numpy as np
import argparse
import os
from glob import glob
import pandas as pd
import torch
import torchaudio
import fairseq
from transformers import AutoFeatureExtractor, Wav2Vec2Processor


class HiddenFeatureExtractor:
    def __init__(self) -> None:
        ##################################################################################################
        ## Important Intializations
        ##################################################################################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_url = "facebook/wav2vec2-base"
        self.model_path = "./model/SPRING_INX_ccc_wav2vec2_SSL.pt"
        self.hiddenFeaturesPath = "./features/features.npy"
        self.cols = np.arange(0,1024,1)
        # self.label_list = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun','tam', 'tel']
        # self.input_column = "path"
        ### Load Model
        self.processor, self.model_wave2vec2 = self.load_model(self.model_path)
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.processor.feature_extractor.return_attention_mask = True ## to return the attention masks
        ### Load VAD Model
        self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
    
    def load_model(self, path: str):
        # Loading ccc-wav2vec model from path
        model_wave2vec2, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        model_wave2vec2 = model_wave2vec2[0].to(self.device)
        model_wave2vec2.eval()
        processor = Wav2Vec2Processor.from_pretrained(self.repo_url)
        return processor, model_wave2vec2


    def load_audio(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        if speech_array.shape[0] > 1:
            # Convert to mono by averaging the channels
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech_array = resampler(speech_array).squeeze(0)
        ### Apply Silero VAD
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self.vad_utils
        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(speech_array, self.vad_model, sampling_rate=self.target_sampling_rate)
        ### When an audio does not have speech
        if len(speech_timestamps) == 0:
            return np.array([])
        else:
            vad_speech = collect_chunks(speech_timestamps, speech_array)
            return np.array(vad_speech)
    
    def preprocess_audio(self, audio_list):
        names = [path.split("/")[-1] for path in audio_list]
        speech_list = [self.load_audio(path) for path in audio_list]
        return names, speech_list

    def hiddenFeatures(self, speech_list):
        
        features = self.processor(speech_list, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.device)
        attention_mask  = features.attention_mask.to(self.device)
        if len(input_values.shape) == 4:
            input_values = input_values.squeeze(1).squeeze(1)  # Remove extra dimensions if present
        
        with torch.no_grad():
            # Pass attention_mask to the model to prevent attending to padded values
            # Check if the model is wrapped in DataParallel
            if isinstance(self.model_wave2vec2, torch.nn.DataParallel):
                hidden_features = self.model_wave2vec2.module.forward(input_values, mask=None, features_only=True, padding_mask=attention_mask)['x']
            else:
                hidden_features = self.model_wave2vec2.forward(input_values, mask=None, features_only=True, padding_mask=attention_mask)['x']
            # ccc-wav2vec embedding (hidden features) of listed audio
            np.save(self.hiddenFeaturesPath, hidden_features.detach().cpu().numpy())
        return hidden_features.detach().cpu().numpy()
    
    
def extractor():
    parser = argparse.ArgumentParser(description='Spoken language identification (WSSL uVector) script with command line options.')

    # Command line options
    parser.add_argument('path', help='Path to file or directory')
    args = parser.parse_args()
    path = args.path

    # Get the list of all files from the path
    file_list = []
    # When path is a file
    if os.path.isfile(path) and path.endswith(".wav"):
        file_list.append(path)
    # When path is a directory
    elif os.path.isdir(path):
        file_list = [file
	                 for path1, subdir, files in os.walk(path)
	                 for file in glob(os.path.join(path1, "*.wav"), recursive=True)]
        if len(file_list) == 0:
            print("Error: {} does not contains wav file.".format(path))
    else:
        print("Error: {} is not a valid file/directory path.".format(path))

    ### Create HiddenFeatureExtractor object
    evaluater = HiddenFeatureExtractor()
    file_names, speech_list = evaluater.preprocess_audio(file_list)
    hidden_features = evaluater.hiddenFeatures(speech_list)


if __name__ == '__main__':
    extractor()
    