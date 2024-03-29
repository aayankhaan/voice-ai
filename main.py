# created by Aayan Khan

import numpy as np
from google.api_core import retry
import speech_recognition as sr
from gtts import gTTS
import pygame
import google.generativeai as genai


@retry.Retry()
def retry_chat(**kwargs):
    return genai.chat(**kwargs)


def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename


def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def listen_question():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please ask your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


genai.configure(api_key="API_KEY") # get api key by https://ai.google.dev/

models = [m for m in genai.list_models() if 'generateMessage' in m.supported_generation_methods]
model = models[0].name 
print("Selected model:", model)

question = listen_question()
print("Your question:", question)

response = retry_chat(
    model=model,
    context="You are an expert at solving word problems.",
    messages=question,
)


print("AI Response:", response.last)


audio_file = text_to_speech(response.last)
play_audio(audio_file)
