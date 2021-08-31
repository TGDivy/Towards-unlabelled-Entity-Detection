import torch

from tqdm.notebook import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

import urllib.request
import random
import pyconll
import pickle

import pandas as pd
import numpy as np
import logging
from pandas import read_pickle

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "start":"<s>",
    "end":"</s>"
}
SPECIAL_TOKENS_VALUES = []

class Dataset(torch.utils.data.Dataset):
    def __init__(self,args, tokenizer, split_type, download=False):
        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenizer
        self.start = SPECIAL_TOKENS["start"]
        self.end = SPECIAL_TOKENS["end"]
        self.split_type = split_type
        if download:
            self.download_files()
            
        self.examples = []
        
        if args.UD:
            sentences = self.read_UD("datasets/en_partut-ud-"+self.split_type+".conllu")
            self._create_examples(sentences, self.F_labels_UD)

        if args.tpkoc:
            sentences = self.read_tpkoc("datasets/tpkoc_"+split_type+".p")
            self._create_examples(sentences, self.F_labels_tpkoc)
            
        if args.conll2003:
            sentences = self.read_conll("datasets/conll_"+split_type+".txt")
            self._create_examples(sentences, self.F_labels_conll)

        if args.GMB:
            sentences = self.read_GMB("datasets/GMB.txt")
            self._create_examples(sentences, self.F_labels_GMB)
    
    def download_files(self,):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu', 'en_partut-ud-dev.conllu')
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu', 'en_partut-ud-test.conllu')
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu', 'en_partut-ud-train.conllu')
    
    def read_conll(self, filepath):
        final = []
        sentences = []

        with open(filepath, 'r') as f:
            for line in f.readlines():

                if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                    if len(sentences) > 0:
                        final.append(sentences)
                        sentences = []
                else:
                    l = line.split(' ')
                    sentences.append((l[0],l[1], l[3].strip('\n')))
        return final
    
    def read_tpkoc(self, path):
        keys = pickle.load(open(path, "rb" ))
        obj = pickle.load(open("datasets/tpkoc.p", "rb"))
        dataset = []
        for key in keys:
            sentence = []
            for (word, tag) in obj[key]:

                if word in [".","?","!"] and len(sentence)>=120:
                    sentence.append((word,tag))
                    dataset.append(sentence)
                    sentence = []
                else:
                    sentence.append((word,tag))
        return dataset
    
    def read_UD(self, path):
        data = pyconll.load_from_file(path)
        tagged_sentences=[]
        t=0
        for sentence in data:
            tagged_sentence=[]
            for token in sentence:
                if token.upos and token.form:
                    t+=1
                    tagged_sentence.append((token.form.lower(), token.upos))
            tagged_sentences.append(tagged_sentence)
        return tagged_sentences
    
    def read_GMB(self, path):
        data = pd.read_csv(path, sep="\t", header=None, encoding="latin1")
        data.columns = data.iloc[0]
        data = data[1:]
        data.columns = ['Index','Sentence #','Word','POS','Tag']
        data = data.reset_index(drop=True)
        n_sent = 1.0
        data = data
        empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                            s["POS"].values.tolist(),
                                                            s["Tag"].values.tolist())]
        grouped = data.groupby("Sentence #").apply(agg_func)
        sentences = [s for s in grouped]
        return sentences
    
    def F_labels_UD(self, group):
        tag = group[1]
        if self.split_type == "train":
            if tag in ["NOUN"]:
                val = np.random.normal(1, scale=0.3)
            elif tag in ["PROPN"]:
                val = np.random.normal(2, scale=0.5)
            else:
                val = 0.0
        else:
            if tag in ["NOUN"]:
                val = 1.0
            elif tag in ["PROPN"]:
                val = 2.0
            else:
                val = 0.0
        return val
    
    def F_labels_tpkoc(self, group):
        tag = group[1]
        if self.split_type == "train":
            if tag in ["NN","NNS"]:
                val = np.random.normal(1, scale=0.45)
            elif tag in ["NNP"]:
                val = np.random.normal(2, scale=0.5)
            else:
                val = 0.0
        else:
            if tag in ["NN","NNS"]:
                val = 1.0
            elif tag in ["NNP"]:
                val = 2.0
            else:
                val = 0.0
        return val
    
    def F_labels_GMB(self, group):
        tag = group[1]
        tag_ner = group[2]

        if tag in ["NN","NNS"]:
            val = np.random.normal(1, scale=0.2)
        elif tag in ["NNP"]:
            val = np.random.normal(1.8, scale=0.15)
        else:
            val = 0.0
        
        if tag_ner not in ["O"]:
            val = np.random.normal(2.8, scale=0.15)

        return val
    
    def F_labels_conll(self, group):
        tag = group[1]
        tag_ner = group[2]

        if self.split_type == "train":
            if tag in ["NN","NNS"]:
                val = np.random.normal(1, scale=0.2)
            elif tag in ["NNP"]:
                val = np.random.normal(1.8, scale=0.15)
            else:
                val = 0.0

            if tag_ner not in ["O"]:
                val = np.random.normal(2.8, scale=0.15)
        else:
            if tag in ["NN","NNS"]:
                val = 1
            elif tag in ["NNP"]:
                val = 2
            else:
                val = 0.0
                
            if tag_ner not in ["O"]:
                val = 3

        return val
    
    def _create_examples(self, sentences, Func_labels):
        logger.info("Creating examples")
        for i, sentence in enumerate(tqdm(sentences)):
            start = 0
            end = 0
            labels = [0]
            tokenized_sentence = [self.start]
            subtoken_map = []    
            for group in sentence: #group= word, pos-tag, NER-tag
                word = group[0]
                if len(tokenized_sentence)<(self.max_seq_length-6):
                    tokenized = self.tokenizer.tokenize(word)
                    end += len(tokenized)
                    subtoken_map.append((start, end))
                    start = end
                    for token in tokenized:
                        tokenized_sentence.append(token)
                        labels.append(Func_labels(group))
            
            labels.append(0)
            tokenized_sentence.append(self.end)
            
            example = {
                "sentence_id":i,
                "tokenized_sentence": tokenized_sentence[:self.max_seq_length-6],
                "labels":labels[:self.max_seq_length-6],
                "subtoken_map":subtoken_map[:self.max_seq_length-6],
            }

            self.examples.append(example)
            
    def __getitem__(self, index):
        example = self.examples[index]
        
        this_inst = {
            "sentence_id": example["sentence_id"],
            "sentence": {
                "input_id": [],
                "input_mask": []                
            },
            "subtoken_map":example["subtoken_map"],
            "tokenized_sentence":example["tokenized_sentence"],            
        }
        
        input_id, input_mask, labels = self.build_input(example["tokenized_sentence"], example["labels"])
        
        this_inst["sentence"]["input_id"] = input_id
        this_inst["sentence"]["input_mask"] = input_mask
        this_inst["label"] = labels

        return this_inst
    
    def build_input(self, sentence, labels):
        sentence  = sentence
        labels = labels
        input_id = self.tokenizer.convert_tokens_to_ids(sentence)
        input_mask = [1]*len(input_id)
        
        while len(input_id) < self.max_seq_length:
            input_id.append(0)
        
        while len(input_mask) < self.max_seq_length:
            input_mask.append(0)
        
        while len(labels) < self.max_seq_length:
            labels.append(0)
            
        assert len(input_id)==self.max_seq_length
        assert len(labels)==self.max_seq_length
        assert len(input_mask)==self.max_seq_length

        return input_id, input_mask, labels

    def collate_fn(self, batch):

        batch_size = len(batch)

        # Dialogue input
        input_ids = [ids["sentence"]["input_id"] for ids in batch]
        input_ids = torch.tensor(input_ids).view(batch_size, -1)
        input_masks = [ids["sentence"]["input_mask"] for ids in batch]
        input_masks = torch.tensor(input_masks).view(batch_size, -1)
        labels = [ids["label"] for ids in batch]
        labels = torch.tensor(labels, dtype=torch.float).view(batch_size, -1)
        
        data_info = {
            "sentence_ids": [ins["sentence_id"] for ins in batch],
            "subtoken_maps":[ins["subtoken_map"] for ins in batch],
            "tokenized_sentences":[ins["tokenized_sentence"] for ins in batch]
        }

        final_input = {
            "inputs":(input_ids, input_masks),
            "labels": labels,
            "data_info":data_info
        }
        
        return final_input
    
    def __len__(self):  
        return len(self.examples)