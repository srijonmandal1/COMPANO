import openai
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np   
import os 
from scipy.io.wavfile import write
from gtts import gTTS

from pydub import AudioSegment
from pydub.playback import play

from twilio.rest import Client

from pymongo import MongoClient

from dotenv import load_dotenv

import sys
import time

load_dotenv()

device_ID = os.getenv('DEVICE_ID')


account_sid = "ACdc3e995b4baec5d0ce6f902600088aac"
auth_token = "c949c558247574a69110fd49299028ed"
client = Client(account_sid, auth_token)


def speak_text(text_msg):
    tts = gTTS(text=text_msg, lang='en')
    tts.save("text.mp3")
    os.system("mpg321 text.mp3")
    os.remove("text.mp3")


def MongoDB_user():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_info = db.userinfo
    return user_info


user_info = MongoDB_user()


def MongoDB_preference():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_preference = db.userpreference
    return user_preference

user_preference = MongoDB_preference()


def MongoDB_events():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_events = db.userevents
    return user_events

user_event = MongoDB_events()

overall_preferences = user_preference.find()

emergency_to_name = user_info.find_one({'device_id':device_ID})["emergency_contact"]

emergency_number = user_info.find_one({'device_id':str(device_ID)})["phone_number"]

associated_name = user_info.find_one({'device_id':device_ID})['name']


def tester():
    sd.default.dtype='int32', 'int32'

    openai.api_key = "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"


    fs = 44100  # Sample rate
    seconds = 7  # Duration of recording


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

        user_input = ""

        try:
            user_input = r.recognize_google(audio)
            # print(f"This is the raw user input:{user_input}")
        except:
            print("entered except")
            if not user_input:
                print("done")
                break
            continue

        busy_flag = False


        if ("call" in str(user_input)) or ("caretaker" in str(user_input)):
            print("I came in here to call")
            initial_txt = f"Ok. I am calling {emergency_to_name} now"
            speak_text(initial_txt)
            message = client.messages.create(
                body=f"Hello {emergency_to_name}.{associated_name} would like to talk to you.",
                to=f"+1{emergency_number}",
                from_="+18333270330"
                )
            next_txt = "How can I assist you?"
            speak_text(next_txt)
            busy_flag = True

        if ("reminder" in str(user_input)) or ("preference" in str(user_input)) or ("reminders" in str(user_input)):
            print("I am telling reminders")
            for each_preference in overall_preferences:
                each_part_time = each_preference['time'].split(':')
                if int(each_part_time[0]) < 12:
                    each_pref_txt = f"{each_preference['text']} at {each_preference['time']} A.M."
                else:
                    each_pref_txt = f"{each_preference['text']} at {each_preference['time']} P.M."
                # each_pref_txt = f"{each_preference['text']} at {each_preference['time']}"
                speak_text(each_pref_txt)
            next_txt = "Is there anything else I can help you with?"
            speak_text(next_txt)
            busy_flag = True

        if ("music" in str(user_input)) or ("play" in str(user_input)):
            initial_txt = "Ok, I am playing some calm music now."
            speak_text(initial_txt)
            # playsound('new_calm.mov')
            song = AudioSegment.from_wav("new_calmer.wav")
            print('playing sound using pydub')
            play(song)
            next_txt = "What else can I help you with?"
            speak_text(next_txt)
            busy_flag = True

        if ("device pause" in str(user_input)):
            # Adjust this
            time.sleep(6)
            break

        if ("device quit" in str(user_input)):
            sys.exit("Device prompted to close")

        if busy_flag == False:
            prompt = user_name + ": " + user_input + "\n" + bot_name+ ": "

            conversation += prompt  # allows for context

            # fetch response from open AI api
            response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=100)
            response_str = response["choices"][0]["text"].replace("\n", "")
            response_str = response_str.split(user_name + ": ", 1)[0].split(bot_name + ": ", 1)[0]

            conversation += response_str + "\n"
            print(response_str)

            speak_text(response_str)
            # engine.runAndWait()

