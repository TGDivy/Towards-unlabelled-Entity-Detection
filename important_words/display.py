import random
from random import randint

import numpy as np
from spacy import displacy


class Display:
    def __init__(
        self,
        minimum_score_to_display=0.5,
        divisions=16,
        categories=["", "NOUN_", "PROPN_", "NER_"],
        predict_mode=False,
    ):
        self.divisions = divisions
        self.categories = categories
        self.n = len(self.categories)
        self.maxscore = self.n - 1
        self.minimum_score_to_display = minimum_score_to_display

        self.predict_mode = predict_mode
        self.label_color = self.color_label_dict()

    def alpha_to_hex(self, alpha):
        return self.rgb_to_hex((0, 0, alpha))

    def rgb_to_hex(self, rgb):
        return "#%02x%02x%02x" % rgb

    def random_hex(self):
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return "#%02x%02x%02x" % rgb

    def color_label_dict(self):
        n_step = self.n / self.divisions
        name = 0
        dic = {}
        self.category_vals = []
        prevn = 0
        previ = 0
        for i in np.linspace(0, self.maxscore, self.divisions):
            name = int(i + 0.5)
            if name != prevn:
                self.category_vals.append(i)
            prevn = name
            previ = i
            colorval = int(255 * (self.maxscore - i) / self.maxscore)
            i = int(round(i, 3) * 100)
            dic[str((i)) + self.categories[int(name)]] = self.rgb_to_hex(
                (colorval, 167, 125)
            )
            if not self.predict_mode:
                dic["XX" + str((i)) + self.categories[int(name)]] = self.rgb_to_hex(
                    (colorval, colorval, 160)
                )
        self.category_vals.append(self.maxscore)
        return dic

    def get_spans(self, tokenizer, tokens, subtoken_map, predictions, scores, labels):
        prevn = 0
        subtoken = []
        text = ""
        spans = []
        start_char = 0

        for (start, end) in subtoken_map:
            token = " " + tokenizer.convert_tokens_to_string(
                tokens[start + 1 : end + 1]
            )
            ent = {"start": start_char, "end": start_char + len(token), "label": None}
            if end - start != 0:
                prediction = sum(predictions[start + 1 : end + 1]) / (end - start)
                label = sum(labels[start + 1 : end + 1]) / (end - start)
                correct = prediction == label
                score = sum(scores[start + 1 : end + 1]) / (end - start)
                score = score.item()
                if score >= self.minimum_score_to_display:
                    ent["label"] = self.score_to_color_label(score, correct)
                    spans.append(ent)
            text += token
            start_char += len(token)
        return text, spans

    def score_to_color_label(self, score, correct):
        v = score
        i = 0
        while v > 0:
            v -= self.maxscore / self.divisions
            i += 1
        category_i = [j for j in range(self.n) if score < self.category_vals[j]][0]
        val = int(round(np.linspace(0, self.maxscore, self.divisions)[i - 1], 3) * 100)
        if correct or self.predict_mode:
            return str(val) + self.categories[category_i]
        else:
            return "XX" + str(val) + self.categories[category_i]

    def render_entities(self, text, spans):
        ent = {
            "text": text,
            "ents": spans,
            "title": None,
        }

        options = {"ents": list(self.label_color.keys()), "colors": self.label_color}
        markdown = displacy.render(ent, manual=True, style="ent", options=options)

        return markdown
