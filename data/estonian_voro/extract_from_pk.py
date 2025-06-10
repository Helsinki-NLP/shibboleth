import sacremoses, string, collections, re

mpn = sacremoses.MosesPunctNormalizer()
mt = sacremoses.MosesTokenizer(lang='et')

url_regex = re.compile(r'(https?):\/\/(www\.)?\w+[\w/\-?=%.]+\.[\w/\-&?=%.]+')
url2_regex = re.compile(r'www\.\w+[\w/\-?=%.]+\.[\w/\-&?=%.]+')
email_regex = re.compile(r'(\w|\.)+@(\w|\.)+\.(\w\w\w?)')


def normalize(s):
	s = mpn.normalize(s.strip())
	s = s.replace("\t", " ")
	s = url_regex.sub('URL', s)
	s = url2_regex.sub('URL', s)
	s = email_regex.sub('EMAIL', s)
	return s

def tokenize(s):
	toklist = mt.tokenize(s)
	ntok = len(toklist)
	tokstr = " ".join(toklist)
	tokstr = tokstr.replace("&quot;", '"').replace("&apos;", "'").replace("&#91;", "[").replace("&#93;", "]").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
	tokstr2, nsub = re.subn(r"(\w) '", r"\1'", tokstr)
	return ntok-nsub, tokstr2


e_data = set()
with open("paralleelkorpus_eesti.txt") as e_file:
	for line in e_file:
		line = normalize(line)
		if line == "URL" or line == "EMAIL":
			continue
		e_data.add(line)

v_data = set()
with open("paralleelkorpus_voro.txt") as v_file:
	for line in v_file:
		line = normalize(line)
		if line == "URL" or line == "EMAIL":
			continue
		v_data.add(line)

print("Estonian:", len(e_data))
print("Voro:", len(v_data))
print("Common:", len(e_data & v_data))
print()

for dupl in list(e_data & v_data):
	e_data.remove(dupl)
	v_data.remove(dupl)

print("Estonian:", len(e_data))
print("Voro:", len(v_data))
print("Common:", len(e_data & v_data))
print()

with open("pkev_all_tok.csv", "w") as outfile:
	for item in e_data:
		ntok, tokstr = tokenize(item)
		outfile.write(f"est\t{item}\t{tokstr}\n")
	for item in v_data:
		ntok, tokstr = tokenize(item)
		outfile.write(f"vro\t{item}\t{tokstr}\n")
