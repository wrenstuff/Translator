import flask
from flask import render_template, redirect, url_for
import german_model_loader 

app = flask.Flask(__name__)
langs = [{'code':'en', 'name':'English'},
{'code':'es', 'name':'Spanish'},
{'code':'de', 'name':'German'},]


@app.route("/")
def home():
    return render_template("home.html", langs=langs)

@app.route("/record")
def record():
    return redirect(url_for("index"))

@app.route("/translate")
def translate():
    return redirect(url_for("index"))

@app.route("/playagain")
def playagain():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)