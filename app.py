import flask
from flask import render_template, redirect, url_for, request
import model_loader as ml
import voice_record as voice

app = flask.Flask(__name__)
starter_langs = [{'code':'en', 'name':'English'},
                 {'code': 'es', 'name':'Spanish'},
                 {'code': 'de', 'name':'German'},]

translated_langs = [{'code':'en', 'name':'English'},
                    {'code': 'es', 'name':'Spanish'},
                    {'code': 'de', 'name':'German'},]

models = {
    "en" : (ml.es_model, ml.es_tokenizer),
    "es" : (ml.es_model, ml.es_tokenizer),
    "de" : (ml.de_model, ml.de_tokenizer),
}

@app.route("/")
def home():
    return render_template("home.html",
                           starter_langs=starter_langs,
                           translated_langs=translated_langs)


@app.route("/record", methods=['POST'])
def record():
    result, recorded_lang = voice.voice_to_text()
    old_text = request.form['text']
    if old_text != "":
        fulltext = old_text + ". " +  result
    else:
        fulltext = result
    return render_template("home.html",
                           text=fulltext, recorded_lang=recorded_lang,
                           starter_langs=starter_langs,
                           translated_langs=translated_langs)


@app.route("/translate", methods=["POST"])
def translate():
    text = request.form['text']
    code = request.form['code']
    model = models.get(code)
    if code not in models:
        translated_text = "Language not supported"
        return translated_text

    model, tokenizer = models[code]
    translated_text = ml.translate_text(model, tokenizer, text)
    return render_template("home.html",
                           text=text, output=translated_text,
                           starter_langs=starter_langs,
                           translated_langs=translated_langs)


@app.route("/playagain")
def playagain():
    return redirect(url_for("static"))

@app.route("/clear")
def clear():
    text = ''
    translated_text = ''
    return render_template("home.html",text=text, output=translated_text,
                           starter_langs=starter_langs,
                           translated_langs=translated_langs)

if __name__ == "__main__":
    app.run(debug=True)