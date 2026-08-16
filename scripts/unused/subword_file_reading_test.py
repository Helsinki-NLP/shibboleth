from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import torch

import pandas as pd
import sys, json

model_type = sys.argv[1]
model_name = sys.argv[2]
test_file = sys.argv[3]
out_file = sys.argv[4]
multilabel = len(sys.argv) > 5 and sys.argv[5] == "-ml"

print(model_type, model_name, test_file, out_file, multilabel) ## AM

device = 0 if torch.cuda.is_available() else -1

test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
test_data["text"] = test_data["text"].str.strip().astype(str)
sentences = list(test_data["text"])
print("Number of sentences:", len(sentences)) ## AM

if multilabel:
    gold_labels = [set([int(y) for y in x.split(",")]) for x in test_data["labels"].astype(str)]
    num_labels = len(set([x for gold_label_set in gold_labels for x in gold_label_set]))
else:
    gold_labels = list(test_data["labels"])
    num_labels = len(set(gold_labels))
print("Number of labels:", num_labels) ## AM

tokenizer = AutoTokenizer.from_pretrained(model_type)
classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

print("Tokenizer loaded on device", device)    
print("Classifier loaded on device", device)
print("Multilabel mode:", multilabel)