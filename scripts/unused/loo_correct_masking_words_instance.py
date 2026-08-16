## the most up-to-date version is attrib_loo.py

from transformers import pipeline
from datasets import Dataset
import torch

import pandas as pd
import sys, json
from tqdm import tqdm

# Expected test file format: [label]\t[raw text]\t[pretokenized text]

model_name = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[3]

in_column = sys.argv[4] # name of the column in which instances are found
# possible values: raw (second column in input file) or tok (third column)

multilabel = len(sys.argv) > 5 and sys.argv[5] == "-ml"

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=model_name, device=device)
print("Pipe loaded on device", device)
print("Multilabel mode:", multilabel)

#test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
# headerless data for BCMS
test_data = pd.read_csv(test_file, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
# assumes same set of labels as training data
labels = sorted(test_data['labels'].unique())
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(id2label)
print(label2id)
test_data["labels"] = test_data["labels"].map(label2id)

try:
    test_data["text"] = test_data[in_column].str.strip().astype(str)
except KeyError:
    sys.exit('Check your in_column argument. Allowed values: tok or raw.')
    
sentences = list(test_data["text"])
if multilabel:
    gold_labels = [set([int(y) for y in x.split(",")]) for x in test_data["labels"].astype(str)]
    gold_labels_str = gold_labels # fixme
    num_labels = len(set([x for gold_label_set in gold_labels for x in gold_label_set]))
else:
    gold_labels = list(test_data["labels"])
    gold_labels_str = [id2label[x] for x in gold_labels]
    #print("Gold labels:", gold_labels)
    num_labels = len(set(gold_labels))

tok_args = {'padding': True, 'truncation': True, 'max_length': pipe.tokenizer.model_max_length}
mask_token = pipe.tokenizer.mask_token

## Aleksandra: add add_prefix_space=True for xlm-r?

def data_iterator():
    for i, row in tqdm(test_data.iterrows()):
        yield row["text"]

print("Predict full sentences")
predictions = pipe(data_iterator(), **tok_args)
pred_labels_scores = [(int(x["label"].replace("LABEL_", "")), x["score"]) for x in predictions]
pred_labels = [x[0] for x in pred_labels_scores]
pred_labels_str = [id2label[x] for x in pred_labels]
pred_scores = [x[1] for x in pred_labels_scores]

# no confusion matrix in multilabel settings
if not multilabel:
    conf_matrix = pd.crosstab(gold_labels_str, pred_labels_str, rownames=['Gold'], colnames=['Predicted'], margins=True)
    print()
    print(conf_matrix)
    print()

print("Predict leave-one-out for correct predictions")
diffs = {}
i = 0
output = []
correct_predictions = 0
for sentence, goldlabel, full_predlabel, full_predscore in tqdm(zip(sentences, gold_labels, pred_labels, pred_scores)):
    #print(sentence)
    # multilabel:  goldlabel is a set => test for inclusion
    # singlelabel: goldlabel is an int => test for equality
    prob_diffs = {} # moving this here to make sure there's output even for incorrect predictions
    if (multilabel and full_predlabel in goldlabel) or (not multilabel and full_predlabel == goldlabel):
        prob_diffs = {'PREDICTION_' : 'correct'}
        correct_predictions += 1

        # difference to initial implementation: if a word occurs several times, remove all its occurrences at once
        # might need to do this on tokenized lowercased text in the future
        words = [x for x in sentence.split(" ") if x != ""]
        unique_words = set(words)
        #print('Unique words: ', len(unique_words))

        # if there is only one word and we remove it, there is nothing left :D
        if len(unique_words) > 1:
            inputs = []
            masked = []
            for w in unique_words:
                input = " ".join([x if x != w else mask_token for x in words])
                inputs.append(input)
                masked.append(w)
            outputs = pipe(inputs, **tok_args)
            #print('Number of tests:', len(outputs))
            
            for loo_word, loo_prediction in zip(masked, outputs):
                loo_predlabel = int(loo_prediction["label"].replace("LABEL_", ""))
                loo_predscore = loo_prediction["score"]
                score_diff = full_predscore - loo_predscore
                #print('Score diff:', score_diff)
                # negative difference means that the left-out word is not at all prototypical for the label, so we just skip this
                # Keeping all diffs that are at least 0.01 ## AM
                if score_diff > 0:
                    prob_diffs[loo_word] = score_diff
                    #print(loo_word, score_diff)

    else:
        prob_diffs = {'PREDICTION_' : 'incorrect'}
        
    output.append(prob_diffs)

print('Correct predictions:', correct_predictions)

with open(out_file, 'w') as of:
    for res in output:
        of.write(json.dumps(res, ensure_ascii=False) + '\n')
