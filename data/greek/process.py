import unicodedata, collections, re

def extract_by_doc(filename):
	with open(filename) as f:
		docs = []
		cur_doc = []
		cur_label = ""
		for line in f:
			if line.startswith(" doc") or line.startswith("doc"):
				if cur_doc != []:
					d = (" ".join(cur_doc), cur_label)
					docs.append(d)
					cur_doc = []
					cur_label = ""
			elif line.startswith("title"):
				continue
			else:
				text, label = line.strip().rsplit(";", 1)
				cur_label = label
				cur_doc.append(text)
		if cur_doc != []:
			d = (" ".join(cur_doc), cur_label)
			docs.append(d)
		for d in docs:
			print(d)


def extract_by_line_filtered(filename, outfilename, skip_first=False):
	remove_boilerplate = 0
	remove_script = 0
	remove_length = 0
	total = 0
	instances = collections.defaultdict(int)
	tokens = collections.defaultdict(int)
	with open(filename, 'r') as f, open(outfilename, 'w') as f2:
		for line in f:
			if skip_first:
				skip_first = False
				continue
			total += 1
			if line.startswith(" doc") or line.startswith("doc") or line.startswith("title"):
				remove_boilerplate += 1
				continue
			if ";" not in line:
				print("Format issue:")
				print(line.strip())
				remove_boilerplate += 1
				continue
			text, label = line.strip().rsplit(";", 1)

			# remove tabs
			text = re.sub(r'(.+)\t\d+$', r'\1', text)
			text = re.sub(r'^\d+\t(.+)$', r'\1', text)
			text = re.sub(r'^[A-ZΑ-Ω ]+\t(.+)$', r'\1', text)
			text = text.replace("\t", " ").strip()

			scripts = [unicodedata.name(x).split(" ")[0] for x in text if x != " "]
			prop = 0 if scripts.count("GREEK") == 0 else scripts.count("GREEK") / len(scripts)
			if prop < 0.75:
				remove_script += 1
				continue
			
			if len(text.split(" ")) < 3:
				remove_length += 1
				continue
			
			instances[label] += 1
			tokens[label] += len(text.split(" "))
			f2.write(f"{label}\t{text}\t\n")
	
	print("Total", total)
	print("Removed boilerplate", remove_boilerplate)
	print("Removed script", remove_script)
	print("Removed length", remove_length)
	print("Instances / Tokens:")
	for c in instances:
		print(c, instances[c], tokens[c])


if __name__ == "__main__":
	#extract_by_doc("subset_5000.csv")
	#extract_by_line_filtered("subset_5000.csv", "subset_5000_filtered.tsv")
	extract_by_line_filtered("all.txt", "subset_all_filtered.tsv", skip_first=True)
