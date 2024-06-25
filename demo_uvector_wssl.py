#####################################################
################# Acknowledgements ##################
## This code is developed under the Natural Language Translation Mission (NLTM) consortium project "Speech Technologies in Indian Languages".
## The copyrights for this code belong to IIT Mandi and IIT Dharwad.
#####################################################
## @author : Sujeet Kumar
#####################################################

################################### How to run the code #####################################
###  python demo_uvector_wssl.py <path_of_an_audio_wav_file>
###  python demo_uvector_wssl.py <path_of_a_directory_containing_audio_wav_files>
#############################################################################################

################## used Library  ############################################################
import torch
import torch.nn as nn
import os
from glob import glob
import numpy as np
import pandas as pd
from torch.autograd import Variable
import sys
import argparse
import matplotlib.pyplot as plt
## Libraries from external python code
from ccc_wav2vec_extractor import HiddenFeatureExtractor


############ number of class and all #####################
Nc = 12 # Number of language classes 
look_back1 = 20 
look_back2 = 50
IP_dim = 1024 # number of input dimension
##########################################

##########################################
#### Function to return processed input data (feature/vector)
def lstm_data(npy_path):
        # X = np.load(npy_path, allow_pickle=True)
        X = npy_path
        Xdata1 = []
        Xdata2 = []
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        np.place(std, std == 0, 1)
        X = (X - mu) / std
        #High resolution low context  
        for i in range(0, len(X) - look_back1, 1):
            a = X[i:(i + look_back1), :]
            Xdata1.append(a)
        Xdata1 = np.array(Xdata1)
        #Low resolution long context 
        for i in range(0, len(X) - look_back2, 2):
            b = X[i + 1:(i + look_back2):3, :]
            Xdata2.append(b)
        Xdata2 = np.array(Xdata2)
        Xdata1 = torch.from_numpy(Xdata1).float()
        Xdata2 = torch.from_numpy(Xdata2).float()
        return Xdata1, Xdata2
###############################################################################

#######################################################
################### uvector Class ####################
# Define the LSTMNet model
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(1024, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * 256, 32, bidirectional=True)
        self.fc_ha = nn.Linear(2 * 32, 100)
        self.fc_1 = nn.Linear(100, 1)
        self.sftmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0)
        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.sftmax(alp)
        T = ht.size(1)
        batch_size = ht.size(0)
        D = ht.size(2)
        c = torch.bmm(al.view(batch_size, 1, T), ht.view(batch_size, T, D))
        c = torch.squeeze(c, 0)
        return c

# Define the MSA_DAT_Net model
class MSA_DAT_Net(nn.Module):
    def __init__(self, model1, model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.att1 = nn.Linear(2 * 32, 100)
        self.att2 = nn.Linear(100, 1)
        self.bsftmax = nn.Softmax(dim=1)
        self.lang_classifier = nn.Linear(2 * 32, Nc, bias=True)

    def forward(self, x1, x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)
        ht_u = torch.cat((u1, u2), dim=0)
        ht_u = torch.unsqueeze(ht_u, 0)
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al = self.bsftmax(alp)
        Tb = ht_u.size(1)
        batch_size = ht_u.size(0)
        D = ht_u.size(2)
        u_vec = torch.bmm(al.view(batch_size, 1, Tb), ht_u.view(batch_size, Tb, D))
        u_vec = torch.squeeze(u_vec, 0)
        lang_output = self.lang_classifier(u_vec)
        return lang_output, u1, u2, u_vec

###############################################################################################
                
######################## Calling uvector ####################
##########################################################
def uvector_wssl(hidden_feature):
    model1 = LSTMNet()
    model2 = LSTMNet()
    model = MSA_DAT_Net(model1, model2)
    ### Load the classification model
    model_path = './model/ZWSSL_train_SpringData_13June2024_e3.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    ### Get the processed input data
    X1, X2 = lstm_data(hidden_feature)
    
    X1 = np.swapaxes(X1, 0, 1)
    X2 = np.swapaxes(X2, 0, 1)
    x1 = Variable(X1, requires_grad=False)
    x2 = Variable(X2, requires_grad=False)
    o1,_,_,_ = model.forward(x1, x2)
    ### Get the probability of all classes and final prediction
    # output = np.argmax(o1.detach().cpu().numpy(), axis=1)
    output =  o1.detach().cpu().numpy()[0]
    pred_all = np.exp(output) / np.sum(np.exp(output))
    Pred = np.argmax(o1.detach().cpu().numpy(), axis=1)
    
    return Pred[0], pred_all


# Function to perform audio feature extraction and language identification using WSSL uVector
def classification_wssl_uvector(hidden_features, file_names):
    pred_labels = []
    for i in range(len(file_names)):
        hidden_feature = hidden_features[i]
        file_name = file_names[i]
        # Perform language identification using uvector models
        lang, prob_all_lang = uvector_wssl(hidden_feature)

        # Define language mappings for display
        lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'pun': 9, 'tam': 10, 'tel': 11}
        id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tam', 11: 'tel'}

        # Get the identified language
        Y1 = id2lang[lang]
        pred_labels.append(Y1)
        # Display pridected language information using a message box
        print("The predicted language of {} audio is {}".format(file_name, Y1))
    
    if len(file_names) == 1:
        # Plot the language identification probabilities
        fig = plt.figure(figsize=(10, 5))
        plt.bar(lang2id.keys(), prob_all_lang, color='maroon', width=0.4)
        plt.yscale("log")
        plt.xlabel("Languages")
        plt.ylabel("Language Identification Probability (in log scale)")
        plt.title("Language Identification Probability of Spoken Audio using WSSL uVector")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    else:
        # Create a DataFrame
        df = pd.DataFrame({'filename': file_names, 'predicted_language': pred_labels})
        # Specify the CSV file path
        csv_file_path = 'predicted_lang.csv'
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print('Data has been saved to {}'.format(csv_file_path))
    return pred_labels


def main():
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

    ### Get ccc-wav2vec features
    ### Create HiddenFeatureExtractor object
    evaluater = HiddenFeatureExtractor()
    # file_names, speech_list = evaluater.preprocess_audio(file_list)
    pred_lang_label, pred_file_list = [], []
    unclassify_file_list, unclassify_label_index = [], []
    for i in range(len(file_list)):
        file_names, speech_list = evaluater.preprocess_audio([file_list[i]])
        # print(speech_list[i].shape)
        if len(speech_list[0]) <= 16400:
            unclassify_file_list.append(file_list[i])   
        else:
            speech = [speech_list[0]]
            hidden_features = evaluater.hiddenFeatures(speech)
            # Call the function for classification
            pred_lang = classification_wssl_uvector(hidden_features, [file_names[0]])
            ### collect the language label and predicted audio file path
            pred_lang_label.append(pred_lang[0])
            pred_file_list.append(file_list[i])
    if len(file_names) > 1:
        # Create a DataFrame
        df = pd.DataFrame({'filename': pred_file_list, 'predicted_language': pred_lang_label})
        # Specify the CSV file path
        csv_file_path = 'predicted_lang.csv'
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print('Data has been saved to {}'.format(csv_file_path))


if __name__ == "__main__":
    main()
    