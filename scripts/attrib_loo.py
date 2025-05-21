# YS April 2025
# Updated version to make it more in line with the SHAP/IG scripts
# - removed multilabel stuff
# - explicitly load model and tokenizer
# - use word boundaries as defined by tokenizer, not whitespace
# - use masking tokens as defined by tokenizer.mask_token
# - use scores of gold label instead of scores for argmax-predicted label (only relevant for incorrect predictions)
# - changed output format

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch

import pandas as pd
import sys, json, argparse, collections
from tqdm import tqdm

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument('checkpoint_path')
cli_parser.add_argument('dataset_path')
cli_parser.add_argument('output_path')
cli_parser.add_argument('instance_type', choices=['raw', 'tok'])
cli_parser.add_argument('method', choices=['token', 'type'])
args = cli_parser.parse_args()

device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
print("Pipe loaded on device", device)

# Expected test file format: [label]\t[raw text]\t[pretokenized text]
test_data = pd.read_csv(args.dataset_path, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
# assumes same set of labels as training data
labels = sorted(test_data['labels'].unique())
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(id2label)
print(label2id)
test_data["labels"] = test_data["labels"].map(label2id)

try:
    test_data["text"] = test_data[args.instance_type].str.strip().astype(str)
except KeyError:
    sys.exit('Check your in_column argument. Allowed values: tok or raw.')

tok_args = {'padding': True, 'truncation': True, 'max_length': pipe.tokenizer.model_max_length}
## Aleksandra: add add_prefix_space=True for xlm-r?

def data_iterator():
    for i, row in tqdm(test_data.iterrows()):
        yield row["text"]

print("Predict full sentences")
predictions = pipe(data_iterator(), top_k=None, **tok_args)
pred_scores = []
correct = 0
for p, gl in zip(predictions, test_data["labels"]):
    pred_score = {}
    max_pred = sorted(p, key=lambda x: x["score"], reverse=True)[0]
    pred_score["max_label_id"] = int(max_pred["label"].replace("LABEL_", ""))
    pred_score["max_label_str"] = id2label[pred_score["max_label_id"]]
    pred_score["max_score"] = sorted(p, key=lambda x: x["score"], reverse=True)[0]["score"]
    pred_score["gold_label_id"] = gl
    pred_score["gold_label_str"] = id2label[gl]
    pred_score["gold_score"] = [x["score"] for x in p if x["label"] == "LABEL_{}".format(gl)][0]
    correct += int(pred_score["max_label_id"] == pred_score["gold_label_id"])
    pred_scores.append(pred_score)

conf_matrix = pd.crosstab([x["gold_label_str"] for x in pred_scores], [x["max_label_str"] for x in pred_scores], rownames=['Gold'], colnames=['Predicted'], margins=True)
print()
print(conf_matrix)
print()
print("Correct predictions:", correct)
print()

# masks one occurrence at a time, will require some aggregation method if there are several occurrences of the same word in a sentence
def mask_words(s):
    new_inputs = []
    masked = []
    inputs = tokenizer(sentence, return_tensors='pt')
    word_ids = torch.tensor([[-1 if idx is None else idx for idx in inputs.word_ids()]])
    for i in range(word_ids.max() + 1):
        masked_input = [
            tokenizer.mask_token if idx == i else tokenizer.decode(inputs.input_ids.masked_select(word_ids == idx))
            for idx in range(word_ids.max() + 1)
        ]
        new_inputs.append(" ".join(masked_input))
        masked.append(tokenizer.decode(inputs.input_ids.masked_select(word_ids == i)))
    return new_inputs, masked

# masks all occurrences of the same token at once
def mask_word_types(s):
    new_inputs = []
    masked = []
    inputs = tokenizer(sentence, return_tensors='pt')
    word_ids = torch.tensor([[-1 if idx is None else idx for idx in inputs.word_ids()]])
    words = [tokenizer.decode(inputs.input_ids.masked_select(word_ids == idx))
            for idx in range(word_ids.max() + 1)]
    unique_words = collections.defaultdict(list)
    for i, w in enumerate(words):
        unique_words[w].append(i)
    for w in unique_words:
        masked_input = [
            tokenizer.mask_token if idx in unique_words[w] else tokenizer.decode(inputs.input_ids.masked_select(word_ids == idx))
            for idx in range(word_ids.max() + 1)
        ]
        new_inputs.append(" ".join(masked_input))
        masked.append(w)
    return new_inputs, masked


print("Predict leave-one-out")
with open(args.output_path, 'w') as of:
    for sentence, full_score in tqdm(zip(test_data["text"], pred_scores)):
        instance_obj = {
            "pred_label": full_score["max_label_str"],
            "gold_label": full_score["gold_label_str"],
            "correct": full_score["max_label_id"] == full_score["gold_label_id"],
            "tokens": [],
            "attribs": []
        }

        if args.method == 'type':
            inputs, masked = mask_word_types(sentence)
        else:
            inputs, masked = mask_words(sentence)
        instance_obj["tokens"] = masked

        # if there is only one word and we remove it, there is nothing left :D
        if len(masked) <= 1:
            instance_obj["attribs"] = [0.0 for x in masked]
        else:
            outputs = pipe(inputs, top_k=None, **tok_args) 
            for loo_prediction in outputs:
                loo_gold_pred = [x["score"] for x in loo_prediction if x["label"] == "LABEL_{}".format(full_score["gold_label_id"])][0]
                score_diff = full_score["gold_score"] - loo_gold_pred
                instance_obj["attribs"].append(score_diff)
        
        of.write(json.dumps(instance_obj, ensure_ascii=False) + '\n')

print("Done")
