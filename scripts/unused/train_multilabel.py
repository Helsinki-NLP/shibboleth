# Work in progress - not sure if we need this at all

import sys
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
import pandas as pd
import torch

model_name = sys.argv[1]
train_file = sys.argv[2]
valid_file = sys.argv[3]
output_dir = sys.argv[4]
column_name = sys.argv[5]

train_df = pd.read_csv(train_file, sep="\t", names=['labels', 'rawtext', 'toktext'], quoting=3)
train_df["text"] = train_df[column_name].str.strip()
train_df = train_df.drop(columns=['rawtext', 'toktext'])
train_df = train_df.head(n=200)
train_ds = Dataset.from_pandas(train_df, split="train")
print("train", len(train_ds))

labels = sorted(list(set(train_df["labels"].str.split(',').explode().tolist())))
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

valid_df = pd.read_csv(valid_file, sep="\t", names=['labels', 'rawtext', 'toktext'], quoting=3)
valid_df["text"] = valid_df[column_name].str.strip()
valid_df = valid_df.drop(columns=['rawtext', 'toktext'])
valid_ds = Dataset.from_pandas(valid_df, split="test")
print("valid", len(valid_ds))

dataset = DatasetDict()
dataset['train'] = train_ds
dataset['test'] = valid_ds

# train_df["text"] = train_df["text"].str.strip()
# train_df["label"] = train_df["labels"].astype(int)
# train_df = train_df.drop(columns=["identifier", "labels"])
# train_df = train_df.dropna()

# valid_df["text"] = valid_df["text"].str.strip()
# valid_df["label"] = valid_df["labels"].astype(int)
# valid_df = valid_df.drop(columns=["identifier", "labels"])
# valid_df = valid_df.dropna()


if "GysBERT" in model_name or "ScandiBERT" in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(instances):
	encoding = tokenizer(instances["text"], padding="max_length", truncation=True)
	labels_batch = [x.split(',') for x in instances["labels"]]
	labels_matrix = np.zeros((len(labels_batch), len(labels)))
	for row_idx, row in enumerate(labels_batch):
		for col_idx, label in id2label.items():
			if label in row:
				labels_matrix[row_idx, col_idx] = 1
	encoding["labels"] = labels_matrix.tolist()
	return encoding

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=len(labels), id2label=id2label, label2id=label2id)

# try to avoid saving errors with fine-tuned Bertic
for param in model.parameters():
	param.data = param.data.contiguous()

training_args = TrainingArguments(
	output_dir=output_dir,
	learning_rate=2e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	num_train_epochs=10,
	weight_decay=0.01,
	eval_strategy="epoch",
	save_strategy="epoch",
	report_to="none",
	load_best_model_at_end=True,
	metric_for_best_model="accuracy"
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
multilabel_threshold = 0.5

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def compute_metrics(eval_pred):
	predictions, labels = eval_pred
	sigmoid = torch.nn.Sigmoid()
	probs = sigmoid(torch.Tensor(predictions))
	print("probs", probs)
	# next, use threshold to turn them into integer predictions
	y_pred = np.zeros(probs.shape)
	y_pred[np.where(probs >= multilabel_threshold)] = 1
	# finally, compute metrics
	y_true = labels
	print("y_pred", y_pred)
	print("y_true", y_true)
	accuracy = accuracy_metric.compute(predictions=y_pred, references=y_true)
	print(accuracy)
	f1_micro_avg = f1_metric.compute(predictions=y_pred, references=y_true, average='micro')
	return {'f1': f1_micro_avg, 'accuracy': accuracy}

# def compute_metrics(eval_pred):
# 	logits, labels = eval_pred
# 	print(logits, labels)
# 	predictions = np.argmax(logits, axis=-1)
# 	print("gold", labels[:20])
# 	print("pred", predictions[:20])
# 	return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
