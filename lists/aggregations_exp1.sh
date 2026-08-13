module load pytorch/2.6


MODELDIR="../models"
OUTDIR='.'

for lnggrp in scandinavian estonian_voro bcms_setimes; do
  if [[ lnggrp == 'bcms_setimes' ]] ; then
    lnggrp_data=cs_setimes/lexicons
  else
    lnggrp_data=$lnggrp
  fi
  mkdir -p $OUTDIR/$lnggrp

  for attrib in ig shap lime loo; do
    for thresh in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
      for mpd in {1..20}; do

        outfile=$OUTDIR/$lnggrp/"list.${lnggrp}.attrib=${attrib}_mpd=${mpd}_thresh=${thresh}_docnorm=F_tfidf=F.csv"
        if [ ! -f $outfile ]; then
          touch $outfile
          python3 ../scripts/aggregate_sacx.py \
            $outfile \
            $(find $MODELDIR/$lnggrp -type f -name ${attrib}_token.jsonl) \
            --max_per_doc $mpd \
            --selection_threshold $thresh \
            --items_in_list 100 \
            --blacklist ../data/$lnggrp_data/filtered_blacklist.txt;
        fi;

        outfile=$OUTDIR/$lnggrp/"list.${lnggrp}.attrib=${attrib}_mpd=${mpd}_thresh=${thresh}_docnorm=F_tfidf=T.csv"
        if [ ! -f $outfile ]; then
          touch $outfile
          python3 ../scripts/aggregate_sacx.py \
            $outfile \
            $(find $MODELDIR/$lnggrp -type f -name ${attrib}_token.jsonl) \
            --tfidf \
            --max_per_doc $mpd \
            --selection_threshold $thresh \
            --items_in_list 100 \
            --blacklist ../data/$lnggrp_data/filtered_blacklist.txt;
        fi;

        outfile=$OUTDIR/$lnggrp/"list.${lnggrp}.attrib=${attrib}_mpd=${mpd}_thresh=${thresh}_docnorm=T_tfidf=F.csv"
        if [ ! -f $outfile ]; then
          touch $outfile
          python3 ../scripts/aggregate_sacx.py \
            $outfile \
            $(find $MODELDIR/$lnggrp -type f -name ${attrib}_token.jsonl) \
            --docnorm \
            --max_per_doc $mpd \
            --selection_threshold $thresh \
            --items_in_list 100 \
            --blacklist ../data/$lnggrp_data/filtered_blacklist.txt;
        fi;

        outfile=$OUTDIR/"list.${lnggrp}.attrib=${attrib}_mpd=${mpd}_thresh=${thresh}_docnorm=T_tfidf=T.csv"
        if [ ! -f $outfile ]; then
          touch $outfile
          python3 ../scripts/aggregate_sacx.py \
            $outfile \
            $(find $MODELDIR/$lnggrp -type f -name ${attrib}_token.jsonl) \
            --docnorm \
            --tfidf \
            --max_per_doc $mpd \
            --selection_threshold $thresh \
            --items_in_list 100 \
            --blacklist ../data/$lnggrp_data/filtered_blacklist.txt;
        fi;
      done
    done
  done
done
