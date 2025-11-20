from faster_whisper import WhisperModel
import speech_recognition as sr
import pyaudio

model = WhisperModel("medium", device="cuda")

def voice_to_text():
    recording = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for Audio...")
        audio = recording.listen(source)

    with open("voice.wav", "wb") as f:
        f.write(audio.get_wav_data())

    segments, info = model.transcribe("voice.wav", beam_size=5, language=None)

    text = "".join([seg.text for seg in segments])
    detected_lang = info.language

    return text.strip(), detected_lang



# result, lang = voice_to_text()