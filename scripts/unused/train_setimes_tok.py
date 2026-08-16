import sys
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
import pandas as pd

model_name = sys.argv[1]
train_file = sys.argv[2]
valid_file = sys.argv[3]
output_dir = sys.argv[4]

#train_df = pd.read_csv(train_file, sep="\t", names=['labels', 'text', 'tok'], quoting=3)
train_df = pd.read_csv(train_file, sep="\t", names=['labels', 'raw', 'text'], quoting=3)
train_df["text"] = train_df["text"].str.strip()
#train_df["label"] = train_df["labels"].astype(int)
train_df["label"] = train_df["labels"].map({'hr' : 0, 'sr' : 1})
#train_df = train_df.drop(columns=["tok", "labels"])
train_df = train_df.drop(columns=["raw", "labels"])
train_df = train_df.dropna()
train_ds = Dataset.from_pandas(train_df, split="train")
print("train", len(train_ds))

#valid_df = pd.read_csv(valid_file, sep="\t", names=['labels', 'text', 'tok'], quoting=3)
valid_df = pd.read_csv(valid_file, sep="\t", names=['labels', 'raw', 'text'], quoting=3)
valid_df["text"] = valid_df["text"].str.strip()
#valid_df["label"] = valid_df["labels"].astype(int)
valid_df["label"] = valid_df["labels"].map({'hr' : 0, 'sr' : 1})
#valid_df = valid_df.drop(columns=["tok", "labels"])
valid_df = valid_df.drop(columns=["raw", "labels"])
valid_df = valid_df.dropna()
valid_ds = Dataset.from_pandas(valid_df, split="test")
print("valid", len(valid_ds))

labels = train_ds.unique('label')
print("labels", labels)

dataset = DatasetDict()
dataset['train'] = train_ds
dataset['test'] = valid_ds

#tokenizer = AutoTokenizer.from_pretrained(model_name)

if "GysBERT" in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(instances):
	return tokenizer(instances["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
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

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	print("gold", labels[:20])
	print("pred", predictions[:20])
	return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
