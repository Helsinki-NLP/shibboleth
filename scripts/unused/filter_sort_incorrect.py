import json, sys, collections
import pandas as pd

def process(model_dir, test_file, topn, near_miss_threshold=0.05):
	data = {}
	no_change_data = []
	n_empty = 0
	n_near_miss = 0
	n_processed = 0

	with open(test_file, 'r') as tf:
		test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
		test_data["text"] = test_data["text"].str.strip().astype(str)

	with open(f"{model_dir}/expl_incorrect.json", 'r') as ef:
		for line in ef:
			wordinfo = json.loads(line)
			if wordinfo["change_words"] == []:
				n_empty += 1
				no_change_data.append((wordinfo["index"], wordinfo["predicted_label"], wordinfo["gold_label"], test_data.iloc[wordinfo["index"]]["text"]))
			elif wordinfo["predicted_score"] - wordinfo["gold_score"] < near_miss_threshold:
				n_near_miss += 1
			else:
				n_processed += 1
				if wordinfo["predicted_label"] not in data:
					data[wordinfo["predicted_label"]] = collections.defaultdict(int)
				for w in wordinfo["change_words"]:
					data[wordinfo["predicted_label"]][w] += 1
	print("Empty:", n_empty)
	print("Near miss:", n_near_miss)
	print("Processed:", n_processed)

	with open(f"{model_dir}/filtered_words_incorrect.txt", "w") as f:
		for cl in sorted(data):
			f.write(f"*** {cl} ***\n")
			sorted_data = sorted(data[cl], key=data[cl].get, reverse=True)
			for word in sorted_data[:topn]:
				f.write(f"{word}\t{data[cl][word]}\n")
			f.write("\n")
	
	with open(f"{model_dir}/incorrect_sentences.txt", "w") as f:
		f.write("index\tpred_label\tgold_label\ttext\n")
		for d in no_change_data:
			f.write(f"{d[0]}\t{d[1]}\t{d[2]}\t{d[3]}\n")


if __name__ == "__main__":
	modeldir = sys.argv[1]
	testfile = sys.argv[2]
	topn = int(sys.argv[3])
	process(modeldir, testfile, topn)
