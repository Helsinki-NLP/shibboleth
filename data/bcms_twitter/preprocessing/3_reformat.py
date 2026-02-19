import sys
import pandas as pd

label_dict = {
	"bs": 0,
	"hr": 1,
	"me": 2,
	"sr": 3,
}

def str_to_int(s):
	return ",".join([str(label_dict[x]) for x in s.split(",")])


if __name__ == "__main__":
	dataid = sys.argv[1]	# e.g. "seg_nourl_equalclasses"
	outid = sys.argv[2]		# e.g. "twitter_eqcl"
	add_news = len(sys.argv) > 3 and (sys.argv[3] == "-add-news")

	for split in ("train", "dev", "test"):
		df = pd.read_csv(f"{split}_{dataid}.tsv", sep="\t", header=None, names=["identifier", "label_str", "text"], quoting=3) 
		print(split, "twitter", dataid, df.shape)
		df["labels"] = df["label_str"].apply(str_to_int)
		df["text"] = df["text"].str.replace(r'^"+', '', regex=True)
		df["text"] = df["text"].str.replace(r'"+$', '', regex=True)
		df["text"] = df["text"].str.replace(r'"+', '"', regex=True)
		df["text"] = df["text"].str.replace(r' +', ' ', regex=True)
		df["text"] = df["text"].str.strip()
		df = df[["labels", "identifier", "text"]]
		
		if split == "train":
			print(split, "single-label", df.shape)
			df.to_csv(f"data_{outid}_{split}_sl.tsv", sep="\t", quoting=3)
		
		else:
			print(split, "multi-label", df.shape)
			df.to_csv(f"data_{outid}_{split}_ml.tsv", sep="\t", quoting=3)

			df["label_list"] = df["labels"].str.split(',')
			df = df.explode("label_list", ignore_index=True)
			df = df[["label_list", "identifier", "text"]].rename(columns={'label_list': 'labels'})
			print(split, "single-label", df.shape)
			df.to_csv(f"data_{outid}_{split}_sl.tsv", sep="\t", quoting=3)
		
		print()
