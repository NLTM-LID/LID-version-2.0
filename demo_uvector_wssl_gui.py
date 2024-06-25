#####################################################
################# Acknowledgements ##################
## This code is developed under the Natural Language Translation Mission (NLTM) consortium project "Speech Technologies in Indian Languages".
## The copyrights for this code belong to IIT Mandi and IIT Dharwad.
#####################################################
## @author : Sujeet Kumar
#####################################################

## Python installed libraries
from tkinter import *
import pygame
from tkinter import filedialog
from tkinter import messagebox
from tkinter.filedialog import askopenfile
import shutil
import random
import matplotlib.pyplot as plt
from datetime import datetime
## Libraries from external python code
from demo_uvector_wssl import *
from sound import *

### Create HiddenFeatureExtractor object
evaluater = HiddenFeatureExtractor()

# Function to perform audio feature extraction and language identification using WSSL uVector
def classification_wssl_uvector_gui():
    # Define supported audio file types
    mask_list = [("Sound files", "*.wav")]
    
    # To get the BNF features of the selected audio file
    selected_audio = filedialog.askopenfilename(initialdir='', filetypes=mask_list)
    if len(selected_audio) > 0:
        ### Get ccc-wav2vec features
        print([selected_audio])
        file_names, speech_list = evaluater.preprocess_audio([selected_audio])
        if len(speech_list[0]) <= 16400:
            print("The audio file does not contain specch")
            lang = 'None'
            messagebox.showinfo("Given audio does not contain speech. It may contain only Noise/Silence")
        else:
            speech = [speech_list[0]]
            hidden_features = evaluater.hiddenFeatures(speech)
            # # Call the function for classification
            # classification_wssl_uvector(hidden_features, selected_audio)

            # Perform language identification using uvector models
            lang, prob_all_lang = uvector_wssl(hidden_features[0])

            # Print language identification results
            # print(prob_all_lang)

            # Define language mappings for display
            lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'pun': 9, 'tam': 10, 'tel': 11}
            id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tam', 11: 'tel'}

            # Get the identified language
            Y1 = id2lang[lang]

            # Display language information using a message box
            # messagebox.showinfo("Given audio language is", Y1)
            answer = messagebox.askyesno(title='Identification Result and Confirmation', message="The predicted language of given audio is \n\n\t\t{}\n\nPlease confirm that predicted language is correct?".format(Y1))
            if answer:
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")+str(random.randint(1,1000))
                new_audio_filename = "{}.wav".format(current_time)
                destpath_audio = './classified_audio/{}/{}'.format(Y1, new_audio_filename)
                shutil.copy(selected_audio, destpath_audio)
            else:
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")+str(random.randint(1,1000))
                new_audio_filename = "{}.wav".format(current_time)
                destpath_audio = './unclassified_audio/{}'.format(new_audio_filename)
                shutil.copy(selected_audio, destpath_audio)

            # Plot the language identification probabilities
            fig = plt.figure(figsize=(10, 5))
            plt.bar(lang2id.keys(), prob_all_lang, color='maroon', width=0.4)
            plt.yscale("log")
            plt.xlabel("Languages")
            plt.ylabel("Language Identification Probability (in log scale)")
            plt.title("Language Identification Probability of Spoken Audio using WSSL uVector")
            plt.show()

# Function to play selected sound file
def play_sound():
    mask_list = [("Sound files", "*.wav")]
    sound_file = filedialog.askopenfilename(initialdir='', filetypes=mask_list)
    if len(sound_file) > 0:
        # Load and play the selected sound file using pygame
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()

# Function to stop the currently playing sound
def stop_sound():
    pygame.mixer.music.stop()

# Function to display the main GUI frame with buttons
def GUI_Frame():
    first_frame.grid_forget()
    first_frame.grid(sticky=N + S + W + E)
    first_frame.configure(background="#D9D8D7")

    # Button to record audio
    record_btn = Button(first_frame, text="Start Recording", command=lambda m=1:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    # Button to stop audio recording
    stop_record_btn = Button(first_frame, text="Stop Recording", command=lambda m=2:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    # Button to play recorded audio
    play_record_btn = Button(first_frame, text="Play Recording", command=lambda m=3:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    # Button to stop playing recorded audio
    stopplay_record_btn = Button(first_frame, text="Stop Playing\nRecording", command=lambda m=4:threading_rec(m),fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    
    # Button to play selected audio file
    play_saved_audio = Button(first_frame, text="Play Saved Audio", command=play_sound, fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    
    # Button to stop playing audio
    stop_saved_audio = Button(first_frame, text="Stop Saved Audio", command=stop_sound, fg="white", bg="OrangeRed4", activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    
    # Button to perform audio prediction and language identification using WSSL uVector
    classify2 = Button(first_frame, text="Identify Language\n(Using WSSL uVector)", fg="white", bg="OrangeRed4", command=classification_wssl_uvector_gui, activeforeground="black", activebackground="coral", relief="raised", bd=8, font = ('', 13, 'bold'))
    
    # Position of above buttons
    record_btn.grid(row=1, column=0, padx=10, pady=15)
    stop_record_btn.grid(row=1, column=2, padx=10, pady=10)
    play_record_btn.grid(row=2, column=0, padx=10, pady=10)
    stopplay_record_btn.grid(row=2, column=2, padx=10, pady=10)
    play_saved_audio.grid(row=3, column=0, padx=10, pady=20)
    stop_saved_audio.grid(row=3, column=2, padx=10, pady=20)
    classify2.grid(row=4, column=0, columnspan=3, padx=10, pady=30)


# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Create the main GUI window
MainFrame = Tk()
MainFrame.geometry("480x360")
MainFrame.title("Indian Spoken Language Identification")
MainFrame.configure(background="#D9D8D7")

# Create the menu bar
menubar = Menu(MainFrame)
menubar.add_command(label="", activebackground="OrangeRed4", activeforeground="black", command=GUI_Frame)
MainFrame.config(menu=menubar)

# Create the first frame
first_frame = Frame(MainFrame, width=480, height=360)
GUI_Frame()

# Start the main loop
MainFrame.mainloop()
