import collections, string

# use the kaikki white/blacklist as starting points and move elements around if there is strong evidence from the SLIDE training set

text_token_counts = collections.defaultdict(dict)
with open("/scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/all_tok.csv") as textfile:
	for line in textfile:
		elements = line.strip().split("\t")
		labels = elements[0].split(",")
		tokens = elements[2].split(" ")
		for t in tokens:
			for l in labels:
				if not l in text_token_counts[t]:
					text_token_counts[t][l] = 0
				text_token_counts[t][l] += 1
print(f"{len(text_token_counts)} tokens loaded from text file")

text_token_langs = {}
for t in text_token_counts:
	new_set = set([l for l in text_token_counts[t] if text_token_counts[t][l] > 1])
	# print(t, text_token_counts[t], new_set)
	if new_set != set():
		text_token_langs[t] = new_set
print(f"{len(text_token_langs)} tokens remaining after hapax filtering")

whitelist = {}
with open("whitelist.txt") as wl:
	for line in wl:
		elements = line.strip().split("\t")
		if len(elements) != 2:
			print("Skip:", elements)
			continue
		token = elements[1]
		whitelist[token] = elements[0]
	print(f"{len(whitelist)} tokens loaded from whitelist")

blacklist = set()
with open("blacklist.txt") as wl:
	for line in wl:
		token = line.strip()
		if token == "":
			continue
		blacklist.add(token)
	print(f"{len(blacklist)} tokens loaded from blacklist")

changes = 0
for token in text_token_langs:
	# not enough evidence for this, even with relatively high frequency threshold
	# if token in blacklist and len(text_token_langs[token]) == 1 and text_token_counts[token]["".join(text_token_langs[token])] >= 10:
	# 	print("Black2white:", token, ",".join(text_token_langs[token]))

	if token in whitelist and ",".join(text_token_langs[token]) != whitelist[token]:
		#print("Mismatching labels", token, ",".join(text_token_langs[token]), whitelist[token])
		blacklist.add(token)
		del whitelist[token]
		changes += 1
print("Changes:", changes)

with open("filtered_whitelist.txt", "w") as wl:
	for t in whitelist:
		wl.write(f"{whitelist[t]}\t{t}\n")

with open("filtered_blacklist.txt", "w") as bl:
	for t in blacklist:
		bl.write(f"{t}\n")

#170892 tokens loaded from text file
#94094 tokens remaining after hapax filtering
#356835 tokens loaded from whitelist
#468137 tokens loaded from blacklist
#Changes: 482
