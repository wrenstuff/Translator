import flask
from flask import render_template, redirect, url_for, jsonify, request
from german_model_loader import translate_text
import voice_record as voice

app = flask.Flask(__name__)
langs = [{'code':'en', 'name':'English'},
{'code':'es', 'name':'Spanish'},
{'code':'de', 'name':'German'},]


@app.route("/")
def home():
    return render_template("home.html", langs=langs)


@app.route("/record")
def record():
    result, recorded_lang = voice.voice_to_text()
    return render_template("home.html", text=result, recorded_lang=recorded_lang, langs=langs)


@app.route("/translate", methods=["POST"])
def translate():
    data = ""

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data["text"]
    output_text = translate_text(input_text)

    return redirect(url_for("static")), jsonify({
        "inputText": input_text,
        "outputText": output_text
    })


@app.route("/playagain")
def playagain():
    return redirect(url_for("static"))

if __name__ == "__main__":
    app.run(debug=True)