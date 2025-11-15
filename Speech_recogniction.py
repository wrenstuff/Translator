import speech_recognition as sr
from langdetect import detect

def voice_to_text():
    recording = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for Audio...")
        audio = recording.listen(source)
        try:
            text = recording.recognize_google(audio)
            detected_lang = detect(text)
            return text, detected_lang
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")

# result, lang = voice_to_text()
# print("Text: ", result)
# print("Language: ", lang)