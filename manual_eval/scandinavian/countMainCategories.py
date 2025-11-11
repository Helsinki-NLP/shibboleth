import pandas as pd

data = {}
for i in range(1, 9):
	listdf = pd.read_csv(f"scandinavian.annotated.{i}.csv")
	c_noshib = (listdf['NotShib']).sum()
	c_lexshib = (listdf['LexShib']).sum()
	c_nonlexshib = (listdf['NonLexShib']).sum()
	c_unk = (listdf['Typo_Unk']).sum()
	row = [c_lexshib, c_nonlexshib, c_noshib, c_unk]
	s = sum(row)
	row.append(s)
	data[i] = row

df = pd.DataFrame.from_dict(data, orient='index', columns=["LexShib", "NonLexShib", "NotShib", "Unk", "Sum"])
print(df)
