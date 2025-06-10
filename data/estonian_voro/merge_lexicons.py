import json, string, sys
from tqdm import tqdm

def getKaikkiList(lang, no_alt=False):
	wordlist = set()
	with open(f"wordlists/kaikki-{lang}.jsonl", 'r') as f:
		for line in tqdm(f):
			data = json.loads(line)
			token = data["word"].strip()
			if token == "":
				continue
			if " " in token:
				continue
			if all([x in string.punctuation for x in token]):
				continue
			if no_alt and "alternative" in data["tags"]:
				#print("Skip:", data)
				continue
			wordlist.add(token)
	print(f"{len(wordlist)} words loaded from file kaikki-{lang}.jsonl")
	return wordlist

def getOtherList(filename):
	wordlist = set()
	with open("wordlists/" + filename, 'r') as f:
		for line in tqdm(f):
			token = line.strip()
			if token == "":
				continue
			if " " in token:
				continue
			if all([x in string.punctuation for x in token]):
				continue
			wordlist.add(token)
	print(f"{len(wordlist)} words loaded from file {filename}")
	return wordlist

def getWordsFromCorpus(label):
	wordlist = set()
	with open(f"pkev_all_tok.csv", "r") as f:
		for line in tqdm(f):
			elements = line.strip().split("\t")
			if elements[0] != label:
				continue
			tokens = elements[2].split(" ")
			lowerTokens = [x.lower() for x in tokens if x == x.lower().capitalize() and x != x.lower()]
			wordlist.update(tokens)
			wordlist.update(lowerTokens)
	print(f"{len(wordlist)} words loaded from corpus samples of language {label}")
	return wordlist

def estonian():
	l1 = getKaikkiList("et", no_alt=True)
	l2 = getOtherList("giella-eesti-all-forms.txt")
	l3 = getOtherList("synaqdict-eesti.txt")
	combined = l1 | l2 | l3
	print(len(combined), "unique words in combined list")
	return combined

def voro():
	l1 = getKaikkiList("vro", no_alt=True)
	l2 = getOtherList("giella-voro-all-forms-filtered.txt")
	l3 = getOtherList("giella-voro-all-forms-filtered-without-apostrophes-etc.txt")
	l4 = getOtherList("synaqdict-voro-orthographic-all-variants.txt")
	combined = l1 | l2 | l3 | l4
	print(len(combined), "unique words in combined list")
	return combined

def make_white_black_lists():
	est_dict = estonian()
	vro_dict = voro()
	est_corpus = getWordsFromCorpus("est")
	vro_corpus = getWordsFromCorpus("vro")

	# blacklist are all words that appear both in an Estonian and a Voro resource, one of them being a dictionary
	blacklist = ((est_dict & vro_dict) | (est_dict & vro_corpus) | (vro_dict & est_corpus)) & (est_corpus | vro_corpus)
	print("Blacklist:", len(blacklist))
	# the Estonian whitelist contains all words of the Estonian dictionary that are not in the blacklist, and is then filtered to include only words that appear in the corpus
	whitelist_est = (est_dict - blacklist) & est_corpus
	print("Whitelist EST:", len(whitelist_est))
	whitelist_vro = (vro_dict - blacklist) & vro_corpus
	print("Whitelist VRO:", len(whitelist_vro))

	with open("filtered_blacklist.txt", "w") as f:
		for w in sorted(blacklist):
			f.write(w + "\n")

	with open("filtered_whitelist.txt", "w") as f:
		for w in sorted(whitelist_est):
			f.write("est\t" + w + "\n")
		for w in sorted(whitelist_vro):
			f.write("vro\t" + w + "\n")

if __name__ == "__main__":
	make_white_black_lists()
