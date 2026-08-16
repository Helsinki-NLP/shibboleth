BASEDIR=../..

echo "estonian_voro"
python3 parse_lists.py \
	$BASEDIR/data/estonian_voro/filtered_whitelist.txt \
	$BASEDIR/data/estonian_voro/filtered_blacklist.txt \
	$BASEDIR/data/estonian_voro/pkev_all_tok.csv \
	estonian_voro_lists_overview.csv \
	$BASEDIR/lists/estonian_voro

echo "bcms_setimes"
python3 parse_lists.py \
	$BASEDIR/data/cs_setimes/lexicons/filtered_whitelist.txt \
	$BASEDIR/data/cs_setimes/lexicons/filtered_blacklist.txt \
	$BASEDIR/data/cs_setimes/all_tok.tsv \
	bcms_setimes_lists_overview.csv \
	$BASEDIR/lists/bcms_setimes

echo "scandinavian"
python3 parse_lists.py \
	$BASEDIR/data/scandinavian/filtered_whitelist.txt \
	$BASEDIR/data/scandinavian/filtered_blacklist.txt \
	$BASEDIR/data/scandinavian/slide_sl_all_tok.csv \
	scandinavian_lists_overview.csv \
	$BASEDIR/lists/scandinavian

echo "done"
