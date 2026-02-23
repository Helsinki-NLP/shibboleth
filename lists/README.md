# Aggregate lists

This directory presents the aggregated lists from the individual attribution runs. Each list is aggregated from the 100 predictions (10 iterations * 10 model folds).

Text from the paper:
> The method involves selecting the top $m$ word tokens, according to their attributed weight, in each instance of $\mathcal{F}_i^j$. If the same word type is selected as an explanation for the same labeled language variety by a sufficiently large proportion $p$ of classifiers, the word is deemed to be a \textbf{stable} explanation. We then select the top 100 words for each language variety, according to their aggregate attributed weight. In our experiments, we consider two aggregation functions $f_\mathrm{agg}$: summing or averaging across the weights of all instances.

TODO: describe which script parameter corresponds to which variable from the paper.

The file `list.bcms_setimes.attrib=ig_mpd=10_thresh=0.1_docnorm=F_tfidf=T.csv` is created as follows:

```
python3 ../scripts/aggregate_sacx.py \
    list.bcms_setimes.attrib=ig_mpd=10_thresh=0.1_docnorm=F_tfidf=T.csv \
    ../models/bcms_setimes/iter*/model_fold*/ig_token.jsonl \
    --blacklist ../data/cs_setimes/lexicons/filtered_blacklist.txt \
    --max_per_doc 10 \
    --selection_threshold 0.1 \
    --tfidf
```

