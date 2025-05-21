import sys, json, argparse, string

import scipy.stats
from sklearn.metrics import roc_auc_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-predictions", help="instance-level predictions in .jsonl format")
parser.add_argument("-groundtruth", help="instance-level ground truth in .jsonl format")
parser.add_argument("-threshold", help="removes all predictions with score lower or equal than `threshold`", type=float, default=0.0)
parser.add_argument("-skip-incorrect", help="does not include incorrect predictions in counts", action="store_true")
parser.add_argument("-v", help="prints instance-wise word sets", action="store_true")
args = parser.parse_args()

def normalize(s):
	s = s.strip(string.punctuation)
	s = s.lower()
	return s

all_tp, all_fp, all_fn, all_tn, n_inst = 0, 0, 0, 0, 0
all_white, all_black = [], []
records = []
with open(args.predictions, 'r') as predfile:
	with open(args.groundtruth) as gtfile:
		for predrow, gtrow in zip(predfile, gtfile):
			p = json.loads(predrow)
			gt = json.loads(gtrow)
			if args.skip_incorrect and p["PREDICTION_"] == "incorrect":
				continue
			predicted_words = set([normalize(x) for x in p if x != "PREDICTION_" and p[x] > args.threshold])
			predicted_words.discard("")
			gt_white = set([normalize(x) for x in gt["white"]])
			gt_white.discard("")
			gt_black = set([normalize(x) for x in gt["black"]])
			gt_black.discard("")
			tp = predicted_words & gt_white
			fp = predicted_words & gt_black
			fn = gt_white - predicted_words
			tn = gt_black - predicted_words
			if args.v:
				print("GT:  ", gt_white)
				print("PRED:", predicted_words)
				print("OVLP:", tp)
				print()
			all_tp += len(tp)
			all_fp += len(fp)
			all_fn += len(fn)
			all_tn += len(tn)
			n_inst += 1
			these_whites =  [p[x] for x in p if x != "PREDICTION_" and x in gt_white]
			these_blacks =  [p[x] for x in p if x != "PREDICTION_" and x in gt_black]
			all_white += these_whites
			all_black += these_blacks
			record = {
				'white': these_whites,
				'black': these_blacks,
				'n_white': len(these_whites),
				'n_black': len(these_blacks),
				'n_items': len(predicted_words),
				'scale': np.mean([abs(p[x]) for x in p if x != "PREDICTION_"]),
			}
			records.append(record)

import pandas as pd
records = pd.DataFrame.from_records(records)

print("Predictions: ", args.predictions)
print("Ground truth:", args.groundtruth)
print("Evaluation instances:", n_inst)
acc = all_tp / (all_tp+all_fp+all_fn+all_tn)
print(f"Accuracy:  {100*acc:.3f}%")
prec = all_tp / (all_tp+all_fp)
print(f"Precision: {100*prec:.3f}%")
rec = all_tp / (all_tp+all_fn)
print(f"Recall:    {100*rec:.3f}%")
print(f"F1-score:  {200*prec*rec/(prec+rec):.3f}%")
U, pval = scipy.stats.mannwhitneyu(all_white, all_black, alternative='greater')
f = U / (len(all_white) * len(all_black))
print(f"U-test:    U={U:.1f} p={pval:.3E} f={f:.3f}")
auroc = roc_auc_score(([1] * len(all_white)) + ([0] * len(all_black)), all_white + all_black)
print(f"AUROC:     {auroc*100:.3f}")

print()


print('hasty stats!')

df = records[~records.scale.isna()]
print(scipy.stats.spearmanr(df.scale, df.n_items))
df = df.reset_index()
df['white_spread'] = df.white.apply(np.mean)
df_ = df[['n_white', 'white_spread', 'n_items']].dropna()
print(scipy.stats.spearmanr(df_.n_white, df_.white_spread))

