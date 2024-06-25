#####################################################
################# Acknowledgements ##################
## This code is developed under the Natural Language Translation Mission (NLTM) consortium project "Speech Technologies in Indian Languages".
## The copyrights for this code belong to IIT Mandi and IIT Dharwad.
#####################################################
## @author : Sujeet Kumar
#####################################################

# Import necessary modules
import sounddevice as sd
import soundfile as sf
from tkinter import *
import queue
import threading
from tkinter import messagebox

# Function to handle threading for recording, stopping, and playing audio
def threading_rec(x):
    if x == 1:
        # If recording is selected, then the thread is activated
        t1 = threading.Thread(target=record_audio)
        t1.start()
    elif x == 2:
        # To stop, set the flag to false
        global recording
        recording = False
        messagebox.showinfo(message="Recording stopped")
        messagebox.showinfo(message="Recorded audio is saved as 'record.wav' in 'recorded_audio' folder")
    elif x == 3:
        # To play a recording, it must exist.
        if file_exists:
            # Read the recording if it exists and play it
            data, fs = sf.read("./recorded_audio/record.wav", dtype='float32')
            sd.play(data, fs, blocking=False)
            # sd.wait()
        else:
            # Display an error if none is found
            messagebox.showerror(message="Record something to play")
    elif x == 4:
        # To Stop playing a recording, it must exist.
        sd.stop()

# Callback function to fit data into the queue
def callback(indata, frames, time, status):
    q.put(indata.copy())

# Recording function
def record_audio():
    # Declare global variables
    global recording
    # Set to True to record
    recording = True  
    global file_exists
    # Create a file to save the audio
    messagebox.showinfo(message="Press OK and Start speaking into the mic")
    with sf.SoundFile("./recorded_audio/record.wav", mode='w', samplerate=8000, channels=1) as file:
        # Create an input stream to record audio without a preset time
        with sd.InputStream(samplerate=8000, channels=1, callback=callback):
            while recording:
                # Set the variable to True to allow playing the audio later
                file_exists = True
                # Write into file
                file.write(q.get())
    

# Function to get the path of the recorded audio file
def result():
    return "./recorded_audio/record.wav"

# Create a queue to contain the audio data
q = queue.Queue()
# Declare variables and initialize them
recording = False
file_exists = False
