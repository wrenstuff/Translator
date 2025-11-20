import flask
from flask import render_template, redirect, url_for, request
import model_loader as ml
import voice_record as voice

app = flask.Flask(__name__)

translated_langs = [{'code': 'en', 'name': 'English'},
                    {'code': 'es', 'name': 'Spanish'},
                    {'code': 'de', 'name': 'German'},]

models = {
    "es" : (ml.es_model, ml.es_tokenizer),
    "de" : (ml.de_model, ml.de_tokenizer),
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

    if recorded_lang == "en" and code != "en":
        # English → OTHER
        model_key = code

    elif recorded_lang != "en" and code == "en":
        # OTHER → English
        model_key = recorded_lang

    else:
        # Non-English → Non-English (not yet supported)
        return render_template("home.html",
                               text=text,
                               output="Language not supported",
                               recorded_lang=recorded_lang,
                               translated_langs=translated_langs)

    if model_key not in models:
        translated_text = "Language not supported"
    else:
        model, tokenizer = models[model_key]
        translated_text = ml.translate_text(model, tokenizer, text)

    return render_template("home.html",
                           text=text,
                           output=translated_text,
                           recorded_lang=recorded_lang,
                           translated_langs=translated_langs)


@app.route("/playagain")
def playagain():
    return redirect(url_for("static"))


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