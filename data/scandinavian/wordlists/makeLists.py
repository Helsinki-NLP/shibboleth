import json, collections, string

def mergeLists(langnames, white_filename, black_filename, no_alt=False):
	words = collections.defaultdict(set)
	for lang in langnames:
		with open(f"kaikki-{lang}.jsonl", 'r') as f:
			for line in f:
				data = json.loads(line)
				token = data["word"]
				if token.strip() == "":
					continue
				if " " in token:
					continue
				if all([x in string.punctuation for x in token]):
					continue
				if no_alt and "alternative" in data["tags"]:
					#print("Skip:", data)
					continue
				words[token].add(lang)
	print(f"Tokens loaded from files: {len(words)}")

	n_white, n_black = 0, 0
	with open(white_filename, "w") as whitelist, open(black_filename, "w") as blacklist:
		for w in sorted(words):
			if len(words[w]) == 1:
				whitelist.write(f"{list(words[w])[0]}\t{w}\n")
				n_white += 1
			else:
				blacklist.write(f"{w}\n")
				n_black += 1
	print(f"Tokens written to whitelist {white_filename}: {n_white}")
	print(f"Tokens written to blacklist {black_filename}: {n_black}")


if __name__ == "__main__":
	mergeLists(("da", "nb", "nn", "sv"), "scand_whitelist.txt", "scand_blacklist.txt")
	mergeLists(("da", "nb", "nn", "sv"), "scand_noalt_whitelist.txt", "scand_noalt_blacklist.txt", no_alt=True)

# Tokens loaded from files: 513576
# Tokens written to whitelist scand_whitelist.txt: 456204
# Tokens written to blacklist scand_blacklist.txt: 57372

# Tokens loaded from files: 505285
# Tokens written to whitelist scand_noalt_whitelist.txt: 449734
# Tokens written to blacklist scand_noalt_blacklist.txt: 55551
