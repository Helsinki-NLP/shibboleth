import sys, json, os, pathlib
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="HuggingFace model ID")
parser.add_argument("-train", help="path to csv training file")
parser.add_argument("-valid", help="path to csv validation file")
parser.add_argument("-outdir", help="path to directory for saved model files")
parser.add_argument("-column", help="'raw' for untokenized data (2nd column), 'tok' for tokenized data (3rd column)")
parser.add_argument("-shuffle", action="store_true")
parser.add_argument("-epochs", type=int, default=10)
args = parser.parse_args()

# previous data formats contained a header row - not supported at the moment
train_df = pd.read_csv(args.train, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
train_df["text"] = train_df[args.column].str.strip()
labels = sorted(train_df['labels'].unique())
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(id2label)
print(label2id)
train_df["label"] = train_df["labels"].map(label2id)
train_df = train_df.drop(columns=["labels", "raw", "tok"])
train_df = train_df.dropna()
train_ds = Dataset.from_pandas(train_df, split="train")
if args.shuffle:
	train_ds = train_ds.shuffle(seed=123)
print("train", len(train_ds))

valid_df = pd.read_csv(args.valid, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
valid_df["text"] = valid_df[args.column].str.strip()
valid_df["label"] = valid_df["labels"].map(label2id)
valid_df = valid_df.drop(columns=["labels", "raw", "tok"])
valid_df = valid_df.dropna()
valid_ds = Dataset.from_pandas(valid_df, split="test")
print("valid", len(valid_ds))

dataset = DatasetDict()
dataset['train'] = train_ds
dataset['test'] = valid_ds

if "GysBERT" in args.model or "ScandiBERT" in args.model or "EstBERT" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=512)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

def tokenize_function(instances):
	return tokenizer(instances["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels))
# try to avoid saving errors with fine-tuned Bertic
if "bertic" in args.model:
	for param in model.parameters():
		param.data = param.data.contiguous()

training_args = TrainingArguments(
	output_dir=args.outdir,
	learning_rate=2e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	num_train_epochs=args.epochs,
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

print("Select best checkpoint")
checkpoints = os.listdir(args.outdir)
checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
best_model_checkpoint = ""
best_metric = 0
for checkpoint in checkpoints:
	if "trainer_state.json" in os.listdir(args.outdir + "/" + checkpoint):
		state = json.load(open(args.outdir + "/" + checkpoint + "/trainer_state.json", "r"))
		if state["best_metric"] > best_metric:
			best_metric = state["best_metric"]
			best_model_checkpoint = state["best_model_checkpoint"].split("/")[-1]
print("Best checkpoint:", best_model_checkpoint, best_metric)
fd = os.open(args.outdir, os.O_RDONLY)
os.symlink(best_model_checkpoint, "best", dir_fd=fd)

for checkpoint in checkpoints:
	if checkpoint != best_checkpoint:
		path = pathlib.Path(checkpoint_folder + "/" + checkpoint)
		path.unlink()
