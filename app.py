import json

import nltk
import requests
import spacy
from flask import Flask, redirect, render_template, request, url_for
from spacy import displacy

from coref import Display
from important_words.main import important_words

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)

m = important_words()
d = Display()
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

raw_text = "Fill me in!"


@app.route("/")
def index():
    return redirect(url_for("extract"))


@app.route("/postmethod", methods=["GET", "POST"])
def get_post_javascript_data():
    jsdata = request.form["javascript_data"]
    print(jsdata)
    return "<p>Hello, World!</p>"


@app.route("/extract", methods=["GET", "POST"])
def extract():
    global raw_text
    minimum_display_score = 80
    html = """<div class="entities" style="line-height: 2.5; direction: ltr"> Fill me in! </div>"""
    if request.method == "POST":
        raw_text = request.form["rawtext"]
        minimum_display_score = int(request.form["minimum_score"])
        m.set_minimum_score_to_display(minimum_display_score / 100)
        html = m.pred(raw_text)

    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)

    return render_template(
        "result.html",
        rawtext=raw_text,
        result=result,
        minimum_score=minimum_display_score,
    )


@app.route("/coref", methods=["GET", "POST"])
def coref():
    global raw_text
    html = """<div class="entities" style="line-height: 2.5; direction: ltr"> Fill me in! </div>"""
    if request.method == "POST":
        raw_text = request.form["rawtext"]
        sentences = tokenizer.tokenize(raw_text)
        corefs = json.loads(
            requests.post("http://localhost:9000/predict", json=sentences).text
        )
        d = Display()
        html = d.run(corefs, sentences)

    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)

    return render_template("coref_result.html", rawtext=raw_text, result=result)


if __name__ == "__main__":
    app.run(debug=True)
