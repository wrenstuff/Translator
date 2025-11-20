from faster_whisper import WhisperModel
import speech_recognition as sr
import torch
import pyaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel("medium", device=device, compute_type=compute_type)

def voice_to_text():
    recording = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for Audio...")
        audio = recording.listen(source)

    with open("voice.wav", "wb") as f:
        f.write(audio.get_wav_data())

    segments, info = model.transcribe(
        "voice.wav",
        language=None,
        task="transcribe",
        beam_size=10,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6]
    )

    text = "".join([seg.text for seg in segments])
    detected_lang = info.language

    return text.strip(), detected_lang



# result, lang = voice_to_text()