import sys, json, argparse, string, collections

import scipy.stats
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-predictions", help="instance-level predictions in .jsonl format")
parser.add_argument("-groundtruth", help="instance-level ground truth in .jsonl format")
parser.add_argument("-threshold", help="removes all predictions with score lower or equal than `threshold`", type=float, default=0.0)
parser.add_argument("-skip-incorrect", help="does not include incorrect predictions in counts", action="store_true")
parser.add_argument("-type-aggregation-func", choices=["max", "sum"], default="sum", help="how to convert word-token attributions to word-type observations")
parser.add_argument("-v", help="prints instance-wise word sets", action="store_true")
parser.add_argument("-use-whitelist", help="restrict analysis to whitelisted words", action="store_true")
args = parser.parse_args()

def normalize(s):
	s = s.strip(string.punctuation)
	s = s.lower()
	return s

all_tp, all_fp, all_fn, all_tn, n_inst = 0, 0, 0, 0, 0
all_white, all_black = [], []
with open(args.predictions, 'r') as predfile:
	with open(args.groundtruth) as gtfile:
		for predrow, gtrow in zip(predfile, gtfile):
			p = json.loads(predrow)
			gt = json.loads(gtrow)
			if args.skip_incorrect and not p["correct"]:
				continue
			# no special treatment for tokens that appear more than once, we keep all occurrences whose attribution score pass the threshold
			predicted_words = set([normalize(t) for t, a in zip(p["tokens"], p["attribs"]) if a > args.threshold])
			predicted_words.discard("")
			gt_black = set([normalize(x) for x in gt["black"]])
			gt_black.discard("")
			if args.use_whitelist:
				gt_white = set([normalize(x) for x in gt["white"]])
			else:
				gt_white = set([normalize(x) for x in p["tokens"]]) - gt_black
			gt_white.discard("")
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
			# all_white += [a for t, a in zip(p["tokens"], p["attribs"]) if a > args.threshold and normalize(t) in gt_white]
			# all_black += [a for t, a in zip(p["tokens"], p["attribs"]) if a > args.threshold and normalize(t) in gt_black]
			do_sumtype = args.type_aggregation_func == 'sum'
			white_inst = collections.defaultdict(float) if do_sumtype else collections.defaultdict(lambda: -float('inf'))
			black_inst = collections.defaultdict(float) if do_sumtype else collections.defaultdict(lambda: -float('inf'))
			for t, a in zip(p["tokens"], p["attribs"]):
				# if a <= args.threshold: continue
				nt = normalize(t)
				if nt in gt_white: white_inst[nt] = (a + white_inst[nt]) if do_sumtype else max(a, white_inst[nt])
				if nt in gt_black: black_inst[nt] = (a + black_inst[nt]) if do_sumtype else max(a, black_inst[nt])
			all_white += list(white_inst.values())
			all_black += list(black_inst.values())


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
