import json
import random

import requests
import seaborn as sns
from spacy import displacy


class Display:
    def __init__(self) -> None:
        pass

    def run(self, dic, text):
        self.categories = list(dic.keys())

        self.label_color = self.color_label_dict()
        text, spans = self.get_spans(text, dic)

        return self.render_entities(text, spans)

    def random_hex(self):
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return "#%02x%02x%02x" % rgb

    def color_label_dict(self):
        dic = {}
        colors = sns.color_palette("Spectral", len(self.categories)).as_hex()

        for i in self.categories:
            dic[i] = colors[int(i)]
        return dic

    def get_spans(self, text, dic):
        sentence_lengths = []
        total = 0
        final_text = " ".join(text)
        for i in text:
            sentence_lengths.append(total)
            total += len(i) + 1
        print(sentence_lengths)
        spans = []
        for cluster in dic.keys():
            for entity in dic[cluster]:
                s_offset = entity["sentence_offset"]
                start = entity["span"][0] + sentence_lengths[s_offset]
                end = entity["span"][1] + sentence_lengths[s_offset]
                ent = {"start": start, "end": end, "label": cluster}
                spans.append(ent)
        spans.sort(key=lambda x: x["start"])
        print(dic)
        print("=" * 50)
        print(spans)
        return final_text, spans

    def render_entities(self, text, spans):
        ent = {
            "text": text,
            "ents": spans,
            "title": None,
        }

        options = {"ents": list(self.label_color.keys()), "colors": self.label_color}
        markdown = displacy.render(ent, manual=True, style="ent", options=options)

        return markdown


if __name__ == "__main__":
    text = ["This is Boris Johnson.", "He is very funny."]
    corefs = json.loads(requests.post("http://localhost:9000/predict", json=text).text)
    d = Display()
    print(d.run(corefs, text))
