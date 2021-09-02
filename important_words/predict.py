import logging

import torch
from transformers import RobertaTokenizer

from important_words.display import Display
from important_words.model import Roberta

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {"start": "<s>", "end": "</s>"}


class Predict:
    def __init__(self, args):

        self.device = args.device
        self.max_seq_length = args.max_seq_length
        self.start = SPECIAL_TOKENS["start"]
        self.end = SPECIAL_TOKENS["end"]

        self.tokenizer = RobertaTokenizer.from_pretrained(
            args.loadmodel_name_path, do_lower_case=True
        )

        model = Roberta.from_pretrained(args.loadmodel_name_path)
        model.to(args.device)
        self.model = model

        self.display = Display(
            minimum_score_to_display=args.minimum_score_to_display,
            divisions=args.divisions,
            categories=args.categories,
            predict_mode=True,
        )

    def load_input(self, sentence):
        logger.info("Creating examples")

        start = 0
        end = 0
        labels = [0]
        tokenized_sentence = [self.start]
        subtoken_map = []
        for word in sentence:  # group= word, pos-tag, NER-tag
            if len(tokenized_sentence) < (self.max_seq_length - 6):
                tokenized = self.tokenizer.tokenize(word)
                end += len(tokenized)
                subtoken_map.append((start, end))
                start = end
                for token in tokenized:
                    tokenized_sentence.append(token)

        tokenized_sentence.append(self.end)
        tokenized_sentence = tokenized_sentence[: self.max_seq_length - 6]
        subtoken_map = subtoken_map[: self.max_seq_length - 6]
        labels = [0] * len(tokenized_sentence)

        input_id, input_mask, labels = self.build_input(tokenized_sentence, labels)

        data_info = {
            "sentence_id": 0,
            "subtoken_maps": subtoken_map,
            "tokenized_sentences": tokenized_sentence,
        }

        final_input = {
            "inputs": (
                torch.tensor(input_id).view(1, -1),
                torch.tensor(input_mask).view(1, -1),
            ),
            "labels": torch.tensor(labels).view(1, -1),
            "data_info": data_info,
        }

        batch = self.to_device(final_input)

        return batch

    def build_input(self, sentence, labels):
        sentence = sentence
        labels = labels
        input_id = self.tokenizer.convert_tokens_to_ids(sentence)
        input_mask = [1] * len(input_id)

        while len(input_id) < self.max_seq_length:
            input_id.append(0)

        while len(input_mask) < self.max_seq_length:
            input_mask.append(0)

        while len(labels) < self.max_seq_length:
            labels.append(0)

        assert len(input_id) == self.max_seq_length
        assert len(labels) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        return input_id, input_mask, labels

    def to_device(self, batch):
        batch["inputs"] = tuple(
            input_tensor.to(self.device) for input_tensor in batch["inputs"]
        )
        batch["labels"] = batch["labels"].to(self.device)

        return batch

    def run_batch_selection_eval(self, batch):
        loss, logits = self.model(batch, train=False)
        prediction = (logits > 0.5) * 1 + (logits > 1.5) * 1 + (logits > 2.5) * 1

        text, spans = self.display.get_spans(
            self.tokenizer,
            batch["data_info"]["tokenized_sentences"],
            batch["data_info"]["subtoken_maps"],
            prediction[0],
            logits[0],
            batch["labels"][0],
        )
        markdown = self.display.render_entities(text, spans)
        return markdown
