import collections, sacremoses
#import sacremoses, string, collections, re

LABELS = {"0": "CH", "1": "AT", "2": "DE_S", "3": "DE_N"}
# label distribution: {'DE_S': 281332, 'DE_N': 130136, 'AT': 17155, 'CH': 30286}
MAX_COUNT=130136
MAX_WORDS=40

mt = sacremoses.MosesTokenizer(lang='de')

counts = collections.defaultdict(int)
truncated = 0
tokencounts = collections.defaultdict(int)
with open("jodel_4class_vardialbased_250611.txt") as infile, open("jodel_all_tok.csv", "w") as outfile:
	for line in infile:
		elements = line.strip().split("\t")
		text, labelint = elements
		if counts[LABELS[labelint]] >= MAX_COUNT:
			continue
		if len(text.split(" ")) > MAX_WORDS:
			truncated += 1
			text = " ".join(text.split(" ")[:MAX_WORDS])
		counts[LABELS[labelint]] += 1
		tok = mt.tokenize(text)
		tokencounts[LABELS[labelint]] += len(tok)
		toktext = " ".join(tok)
		outfile.write(f"{LABELS[labelint]}\t{text}\t{toktext}\n")

print("Truncated:", truncated, "/", sum(counts.values()))
print("Instance counts:", counts)
print("Token counts:", tokencounts)
