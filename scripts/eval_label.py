import json, argparse, torch, transformers
import pandas as pd
from sklearn.metrics import classification_report

@torch.no_grad()
def main(args):
    data = pd.read_csv(args.dataset_path, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
    labels = sorted(data['labels'].unique())
    label2id, id2label = {}, {}
    with open(f"{args.checkpoint_path}/../labels.json") as labelfile:
        label2id = json.load(labelfile)
        id2label = {label2id[label]: label for label in label2id}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path).eval().to(device)
    pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, batch_size=32)

    data['predictions'] = pred(data[args.instance_type].to_list())
    data['predictions'] = data['predictions'].apply(lambda x: id2label[int(x['label'].replace("LABEL_", ""))])
    report = classification_report(data['labels'], data['predictions'], digits=4)
    conf_matrix = pd.crosstab(data['labels'], data['predictions'], rownames=['Gold'], colnames=['Predicted'], margins=True)

    with open(args.output_path, 'w') as resultfile:
        resultfile.write("Classification report:\n")
        resultfile.write(report)
        resultfile.write("\n")
        resultfile.write("Confusion matrix:\n")
        resultfile.write(conf_matrix.to_string())
        resultfile.write("\n")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('checkpoint_path')
    cli_parser.add_argument('dataset_path')
    cli_parser.add_argument('output_path')
    cli_parser.add_argument('instance_type', choices=['raw', 'tok'])
    args = cli_parser.parse_args()
    main(args)
