import json, argparse

def load_lists(whitelist_file, blacklist_file):
	whitelist = {}
	with open(whitelist_file, "r") as wl:
		for line in wl:
			elements = line.strip().split("\t")
			if len(elements) != 2:
				continue
			whitelist[elements[1]] = elements[0]
	print(f"Whitelist loaded from {whitelist_file} -- {len(whitelist)} items")

	blacklist = set()
	with open(blacklist_file, "r") as bl:
		for line in bl:
			if line.strip() != "":
				blacklist.add(line.strip())
	print(f"Blacklist loaded from {blacklist_file} -- {len(blacklist)} items")
	return whitelist, blacklist

def annotate_file(text_file, out_file, whitelist, blacklist, verbose):
	print(f"{text_file} ==> {out_file}")
	with open(text_file, "r") as tf, open(out_file, "w") as of:
		for line in tf:
			white_tokens = set()
			black_tokens = set()
			elements = line.strip().split("\t")
			label = elements[0]
			tokens = elements[2].split(" ")
			for t in tokens:
				if t in whitelist:
					if whitelist[t] == label:
						white_tokens.add(t)
					elif verbose:
						print(f"Skipping due to label mismatch: {t} - Whitelist: {whitelist[t]} - Corpus: {label}")
				elif t in blacklist:
					black_tokens.add(t)
				elif t.lower() in whitelist:
					if whitelist[t.lower()] == label:
						white_tokens.add(t)
					elif verbose:
						print(f"Skipping due to label mismatch: {t} - Whitelist: {whitelist[t.lower()]} - Corpus: {label}")
				elif t.lower() in blacklist:
					black_tokens.add(t)
			
			data = {"label": label, "white": list(white_tokens), "black": list(black_tokens)}
			of.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
	cli_parser = argparse.ArgumentParser()
	cli_parser.add_argument('-wl', help="Whitelist path")
	cli_parser.add_argument('-bl', help="Blacklist path")
	cli_parser.add_argument('-i', help="Input CSV file, one instance per line")
	cli_parser.add_argument('-o', help="Output JSONL file, one instance per line")
	cli_parser.add_argument('-v', action="store_true", help="Verbose output")
	args = cli_parser.parse_args()
	wl, bl = load_lists(args.wl, args.bl)
	annotate_file(args.i, args.o, wl, bl, args.v)
