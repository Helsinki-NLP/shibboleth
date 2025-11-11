import pandas as pd

def x_to_bool(s):
	return s == "x"

annotations = pd.read_csv("scandinavian_unique_annotated.csv",
						  converters={"not a shib": x_to_bool, "not minimal": x_to_bool, "regular morph": x_to_bool, "regular phon/spell": x_to_bool, "NE": x_to_bool, "typo/tok/unk": x_to_bool, "lexical": x_to_bool, "counterex": x_to_bool})
annotations = annotations.drop(columns=["Unnamed: 4", "Unnamed: 9", "mark", "count"])
annotations = annotations.rename(columns={"not a shib": "NotShib", "not minimal": "NotMinimal", "regular morph": "Morph", "regular phon/spell": "Phon_Spell", "typo/tok/unk": "Typo_Unk", "lexical": "LexShib", "counterex": "Counterex"})
annotations = annotations.fillna("")
annotations["NonLexShib"] = ~(annotations["NotShib"] | annotations["LexShib"] | annotations["Typo_Unk"])

for i in range(1, 9):
	print(i)
	listdf = pd.read_csv(f"scandinavian.list.{i}.txt")
	joindf = pd.merge(listdf, annotations, on=["token", "label"], suffixes=(None, "_y"))
	joindf = joindf.drop(columns=["in_blacklist_y"])
	joindf.to_csv(f"scandinavian.annotated.{i}.csv", index_label="id")
