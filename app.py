import flask
from flask import render_template, redirect, url_for, request
import model_loader as ml
import voice_record as voice
from gtts import gTTS
import uuid
import os

app = flask.Flask(__name__)

translated_langs = [{'code': 'en', 'name': 'English'},
                    {'code': 'es', 'name': 'Spanish'},
                    {'code': 'de', 'name': 'German'},]

models = {
    "es-en": (ml.es_en_model, ml.es_en_tokenizer),
    "de-en": (ml.de_en_model, ml.de_en_tokenizer),
    "en-es" : (ml.en_es_model, ml.en_es_tokenizer),
    "en-de" : (ml.en_de_model, ml.en_de_tokenizer),
}

@app.route("/")
def home():
    lang = ""
    return render_template("home.html",
                           recorded_lang=lang,
                           translated_langs=translated_langs)


@app.route("/record", methods=['POST'])
def record():
    result, recorded_lang = voice.voice_to_text()
    old_text = request.form['text']

    if old_text != "":
        fulltext = old_text + ". " +  result
    else:
        fulltext = result

    return (render_template("home.html",
                           text=fulltext,
                           recorded_lang=recorded_lang,
                           translated_langs=translated_langs), recorded_lang)


@app.route("/translate", methods=["POST"])
def translate():
    text = request.form['text']
    code = request.form['code']
    recorded_lang = request.form['recorded_lang']

    if recorded_lang == code:
        return render_template("home.html",
                               text=text,
                               output=text,
                               recorded_lang=recorded_lang,
                               translated_langs=translated_langs)

    model_key = f"{recorded_lang}-{code}"

    if model_key not in models:
        return render_template("home.html",
                               text=text,
                               output="Language not supported",
                               recorded_lang=recorded_lang,
                               translated_langs=translated_langs)



    model, tokenizer = models[model_key]

    translated_text = ml.translate_text(model, tokenizer, text)

    tts_path = None
    if code != "en":
        tts_lang = code
    else:
        tts_lang = "en"

    audio_filename = f"tts_{uuid.uuid4().hex}.mp3"

    os.makedirs("static/audio", exist_ok=True)

    tts = gTTS(text=translated_text, lang=tts_lang)
    tts.save("static/audio/" + audio_filename)

    return render_template("home.html",
                           text=text,
                           output=translated_text,
                           tts_file=audio_filename,
                           recorded_lang=recorded_lang,
                           translated_langs=translated_langs)


@app.route("/clear")
def clear():
    text = ''
    translated_text = ''
    return render_template("home.html",
                           text=text,
                           output=translated_text,
                           translated_langs=translated_langs)


if __name__ == "__main__":
    app.run(debug=True)