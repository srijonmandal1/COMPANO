import speech_recognition as sr
import sounddevice as sd
import numpy as np   
import os 
from scipy.io.wavfile import write

sd.default.dtype='int32', 'int32'

fs = 44100  # Sample rate
seconds = 8  # Duration of recording

 # Save as WAV file in 16-bit format
r = sr.Recognizer()

while True:
    print("Start recording the answer.....")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording.astype(np.int32)) 
    sound = "output.wav"

    with sr.AudioFile(sound) as source:
        print("\nlistening...")
        r.adjust_for_ambient_noise(source, duration=0.2)
        r.dynamic_energy_threshold = 3000
        audio = r.listen(source)
    print("no longer listening.\n")

    user_input = r.recognize_google(audio)
    print(user_input)
