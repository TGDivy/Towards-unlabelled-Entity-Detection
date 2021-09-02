import json

import spacy
from flask import Flask, render_template, request, url_for
from spacy import displacy

from important_words.main import important_words

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)


# def analyze_text(text):
# 	return nlp(text)


@app.route("/")
def index():
    # raw_text = "Bill Gates is An American Computer Scientist since 1986"
    # docx = nlp(raw_text)
    # html = displacy.render(docx,style="ent")
    # html = html.replace("\n\n","\n")
    # result = HTML_WRAPPER.format(html)

    return render_template("index.html")


@app.route("/extract", methods=["GET", "POST"])
def extract():
    if request.method == "POST":
        raw_text = request.form["rawtext"]
        m = important_words()
        html = m.pred(raw_text)
        html = html.replace("\n\n", "\n")
        result = HTML_WRAPPER.format(html)

    return render_template("result.html", rawtext=raw_text, result=result)


@app.route("/previewer")
def previewer():
    return render_template("previewer.html")


@app.route("/preview", methods=["GET", "POST"])
def preview():
    if request.method == "POST":
        newtext = request.form["newtext"]
        result = newtext

    return render_template("preview.html", newtext=newtext, result=result)


if __name__ == "__main__":
    app.run(debug=True)
