import logging
import os

from dotmap import DotMap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import nltk.data
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaModel, RobertaTokenizer

nltk.download("punkt")

import json

from important_words.predict import Predict

logger = logging.getLogger(__name__)


class important_words:
    def __init__(self) -> None:
        args = json.load(open("important_words/args.json", "r"))
        self.args = DotMap(args)

    def pred(self, MAIN_DOC):
        self.args.minimum_score_to_display = 0.2
        print("Predicting!\n" + "*" * 75)
        #######################################
        p = Predict(self.args)
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        inputs = tokenizer.tokenize(MAIN_DOC)
        output = ""
        for sentence in inputs:
            if sentence not in [""]:
                words = nltk.word_tokenize(sentence)
                inp = p.load_input(words)
                markdown = p.run_batch_selection_eval(inp)
                output = output + markdown
        return output


if __name__ == "__main__":
    m = important_words()
    doc1 = "This is sentence. It's good enough for now! But what can we really do?"
    m.pred(doc1)

    #######################################
