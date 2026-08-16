# Results of the manual annotations

For each language group, 8 lists were manually annotated. The lists were randomized so as not to influence the annotators. The `randomized_list` folder presents the correspondences, as shown in the following example (`randomized_lists.bcms_setimes.txt`):

```
lists/bcms_setimes/list.bcms_setimes.attrib=loo_mpd=16_thresh=0.6_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=loo_mpd=16_thresh=0.1_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=shap_mpd=16_thresh=0.6_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=shap_mpd=16_thresh=0.1_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=ig_mpd=16_thresh=0.1_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=lime_mpd=16_thresh=0.6_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=lime_mpd=16_thresh=0.1_docnorm=T_tfidf=F.csv
lists/bcms_setimes/list.bcms_setimes.attrib=ig_mpd=16_thresh=0.6_docnorm=T_tfidf=F.csv
````

The first line lists the file corresponding to `bcms_setimes/list1.csv`, the second line the file corresponding to `bcms_setimes/list2.csv`, and so on.

All manually annotated files are provided in CSV format, but the exact structure of the files differ slightly across language groups: different languages show different types of phenomena and different annotators used different strategies to produce the annotations. The scripts in `analyses/shib-eval-contents` provide unified formats of the annotations.
