import pandas as pd

data = {}
langs = ['da', 'nb', 'nn', 'sv']

for i in range(1, 9):
	listdf = pd.read_csv(f"scandinavian.annotated.{i}.csv")
	c = listdf.groupby("label").NE.value_counts()
	row = []
	for label in ('da', 'nb', 'nn', 'sv'):
		try:
			row.append(c.loc[label, True])
		except KeyError:
			row.append(0)
	s = sum(row)
	row.append(s)
	data[i] = row

df = pd.DataFrame.from_dict(data, orient='index', columns=langs + ["sum"])
print(df)

#    da  nb  nn  sv  sum
# 1   4   3   6   1   14
# 2   3   9   9   2   23
# 3   1   5   5   0   11
# 4   1   0   2   0    3
# 5   4   6   6   1   17
# 6   3   1   1   0    5
# 7   1   1   2   1    5
# 8   1   1   2   0    4
