from transformers import pipeline
from datasets import Dataset
import torch

import pandas as pd
import sys, json

model_name = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[3]

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=model_name, device=device)
print("Pipe loaded on device", device)

test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
test_data["text"] = test_data["text"].str.strip().astype(str)
sentences = list(test_data["text"])
gold_labels = list(test_data["labels"])

tok_args = {'padding': True, 'truncation': True, 'max_length': pipe.tokenizer.model_max_length}

def data_iterator():
    for i, row in test_data.iterrows():
        yield row["text"]

print("Predict full sentences")
predictions = pipe(data_iterator(), return_all_scores=True, **tok_args)

print("Predict leave-one-out for incorrect predictions")
outfile = open(out_file, 'w')
pred_labels = []	# for confusion matrix
i = -1
for sentence, gold_label, prediction in zip(sentences, gold_labels, predictions):
	gold_score, pred_score, pred_label = 0, 0, ""
	for p in prediction:
		p_label = int(p["label"].replace("LABEL_", ""))
		p_score = p["score"]
		if p_label == gold_label:
			gold_score = p_score
		if p_score > pred_score:
			pred_score = p_score
			pred_label = p_label
	i += 1
	pred_labels.append(pred_label)

	if gold_label != pred_label:
		datapoint = {"index": i, "gold_label": gold_label, "gold_score": gold_score, "predicted_label": pred_label, "predicted_score": pred_score, "change_words": []}

		# difference to initial implementation: if a word occurs several times, remove all its occurrences at once
		# might need to do this on tokenized lowercased text in the future
		words = [x for x in sentence.split(" ") if x != ""]
		unique_words = set(words)

		# if there is only one word and we remove it, there is nothing left :D
		if len(unique_words) > 1:
			loo_inputs = []
			loo_leftout = []
			for w in unique_words:
				rest = " ".join([x for x in words if x != w])
				loo_inputs.append(rest)
				loo_leftout.append(w)
			loo_outputs = pipe(loo_inputs, **tok_args)

			for loo_word, loo_prediction in zip(loo_leftout, loo_outputs):
				loo_class = int(loo_prediction["label"].replace("LABEL_", ""))
				loo_score = loo_prediction["score"]
				if loo_class == gold_label:
					# these words are indicative of the predicted class, not of the gold class
					datapoint["change_words"].append(loo_word)
		outfile.write(json.dumps(datapoint) + "\n")
outfile.close()
print()

conf_matrix = pd.crosstab(gold_labels, pred_labels, rownames=['Gold'], colnames=['Predicted'], margins=True)
print(conf_matrix)
print()

print("Finished")