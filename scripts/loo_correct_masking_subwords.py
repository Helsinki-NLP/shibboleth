print("Start Python", flush=True)
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
print("Transformers loaded", flush=True)
from datasets import Dataset
import torch

import pandas as pd
import sys, json
print("Libraries loaded", flush=True)

model_type = sys.argv[1]
model_name = sys.argv[2]
test_file = sys.argv[3]
out_file = sys.argv[4]
multilabel = len(sys.argv) > 5 and sys.argv[5] == "-ml"
print("Script arguments:", model_type, model_name, test_file, out_file, multilabel, flush=True) ## AM

device = 0 if torch.cuda.is_available() else -1

test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
test_data["text"] = test_data["text"].str.strip().astype(str)
sentences = list(test_data["text"])
print("Number of sentences:", len(sentences), flush=True) ## AM

if multilabel:
    gold_labels = [set([int(y) for y in x.split(",")]) for x in test_data["labels"].astype(str)]
    num_labels = len(set([x for gold_label_set in gold_labels for x in gold_label_set]))
else:
    gold_labels = list(test_data["labels"])
    num_labels = len(set(gold_labels))
print("Number of labels:", num_labels, flush=True) ## AM

tokenizer = AutoTokenizer.from_pretrained(model_type)
classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

print("Tokenizer loaded on device", device, flush=True)    
print("Classifier loaded on device", device, flush=True)
print("Multilabel mode:", multilabel, flush=True)


def classify(tens):
    with torch.no_grad():
        logits = classifier.forward(input_ids=tens).logits
        predicted_class_id = logits.argmax().item() ## AM: not sure this works for multilabel
        if multilabel:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.nn.functional.softmax(logits, dim=1)
        proba = probs[0][predicted_class_id].item()
        return predicted_class_id, proba
        

# find mask token ID
mask_tok = '<mask>' if model_type == 'xlm-roberta-base' else '[MASK]'
encoding = tokenizer(mask_tok)
if len(encoding['input_ids']) == 3: 
    mask_id = encoding['input_ids'][1]
else:
    sys.exit("Mask token id not found")


tokenized_sent = []
pred_labels = []
pred_scores = []

print("Predict full sentences", flush=True)
tok_args = {'padding': True, 'truncation': True, 'max_length': tokenizer.model_max_length}
## AM: add add_prefix_space=True for xlm-r? Recommended for Roberta


for s in sentences:
    input_ids = tokenizer(s, **tok_args)["input_ids"]
    tens = torch.tensor([input_ids])
    tokenized_sent.append(input_ids)
    pred_label, pred_score = classify(tens)
    pred_labels.append(pred_label)
    pred_scores.append(pred_score)
    


# no confusion matrix in multilabel settings
if not multilabel:
    conf_matrix = pd.crosstab(gold_labels, pred_labels, rownames=['Gold'], colnames=['Predicted'], margins=True)
    print('', flush=True)
    print(conf_matrix, flush=True)
    print('', flush=True)

print("Predict leave-one-out for correct predictions", flush=True)
diffs = {}
i = 0
for sentence, goldlabel, full_predlabel, full_predscore in zip(tokenized_sent, gold_labels, pred_labels, pred_scores):
    # multilabel:  goldlabel is a set => test for inclusion
    # singlelabel: goldlabel is an int => test for equality
    if (multilabel and full_predlabel in goldlabel) or (not multilabel and full_predlabel == goldlabel): 
        prob_diffs = {}
        inputs = []
        masked = []
        for i in range(0, len(sentence)):
            masked.append(sentence[i])
            masked_input = sentence.copy()
            masked_input[i] = mask_id
            inputs.append(masked_input)
            #print(tokenizer.decode(masked_input))
            
        
        for i, loo_masked in zip(inputs, masked):
            tens = torch.tensor([i])
            loo_label, loo_predscore = classify(tens)
            score_diff = full_predscore - loo_predscore
            loo_masked_token = tokenizer.decode(loo_masked)
            if score_diff > 0:
                try:
                    prob_diffs[loo_masked_token].append(score_diff)
                except KeyError:
                    prob_diffs[loo_masked_token] = [score_diff]

    
        # Sort words by probability difference and select top five
        prob_diffs_avg = {}
        for k in prob_diffs:
            avg = sum(prob_diffs[k]) / len(prob_diffs[k])
            prob_diffs_avg[k] = avg
            
        top_words = sorted(prob_diffs_avg.items(), key=lambda x: x[1], reverse=True)[:5]
        for w, sc in top_words:
            if w not in diffs:
                diffs[w] = [[] for x in range(num_labels)]
            diffs[w][full_predlabel].append(sc) 
                

outfile = open(out_file, 'w')
for word in sorted(diffs):
    obj = {"word": word}
    for i in range(num_labels):
        obj[f"cl{i}"] = diffs[word][i]
    outfile.write(json.dumps(obj) + "\n")
outfile.close()
