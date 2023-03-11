import openai
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np   
import os 
from scipy.io.wavfile import write

sd.default.dtype='int32', 'int32'

openai.api_key = "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"

engine = pyttsx3.init()

# not working 
# voices = engine.getProperty('voices')
# engine.setProperty('voice',voices[11].id)

fs = 44100  # Sample rate
seconds = 8  # Duration of recording

newVoiceRate = 165
engine.setProperty('rate',newVoiceRate)


 # Save as WAV file in 16-bit format
r = sr.Recognizer()

conversation = ""
user_name = "You"
bot_name = "Jarvis"

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

    try:
        user_input = r.recognize_google(audio)
    except:
        continue

    prompt = user_name + ": " + user_input + "\n" + bot_name+ ": "

    conversation += prompt  # allows for context

    # fetch response from open AI api
    response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=100)
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(user_name + ": ", 1)[0].split(bot_name + ": ", 1)[0]

    conversation += response_str + "\n"
    print(response_str)

    engine.say(response_str)
    engine.runAndWait()