for attrib in ig loo lime shap; do
  for thresh in 0.1 0.6; do
    if [ ! -f lists/greek/list.greek.attrib\=${attrib}_mpd\=11_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv ]; then
      python3 shibboleth/scripts/aggregate_sacx.py \
        --docnorm \
        --items_in_list 100 \
        --max_per_doc 11 \
        --selection_threshold $thresh \
        lists/greek/list.greek.attrib\=${attrib}_mpd\=11_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv \
        $(find /scratch/project_2006235/shibboleth/models/greek/ -type f -name ${attrib}_token.jsonl ) ;
    fi

    if [ ! -f lists/jodel/list.jodel.attrib\=${attrib}_mpd\=28_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv ]; then
      python3 shibboleth/scripts/aggregate_sacx.py \
        --docnorm \
        --items_in_list 100 \
        --max_per_doc 28 \
        --selection_threshold $thresh \
        lists/jodel/list.jodel.attrib\=${attrib}_mpd\=28_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv \
        $(find /scratch/project_2006235/shibboleth/models/jodel/ -type f -name ${attrib}_token.jsonl ) ;
    fi

    if [ ! -f lists/middle_low_saxon/list.middle_low_saxon.attrib\=${attrib}_mpd\=5_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv ]; then
      python3 shibboleth/scripts/aggregate_sacx.py \
        --docnorm \
        --items_in_list 100 \
        --max_per_doc 5 \
        --selection_threshold $thresh \
        lists/middle_low_saxon/list.middle_low_saxon.attrib\=${attrib}_mpd\=5_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv \
        $(find /scratch/project_2006235/shibboleth/models/middle_low_saxon/ -type f -name ${attrib}_token.jsonl ) ;
    fi

    if [ ! -f lists/bcms_twitter/list.bcms_twitter.attrib\=${attrib}_mpd\=7_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv ]; then
      python3 shibboleth/scripts/aggregate_sacx.py \
        --docnorm \
        --items_in_list 100 \
        --max_per_doc 7 \
        --selection_threshold $thresh \
        lists/bcms_twitter/list.bcms_twitter.attrib\=${attrib}_mpd\=7_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv \
        $(find /scratch/project_2006235/shibboleth/models/bcms_twitter/ -type f -name ${attrib}_token.jsonl | grep -v 'iter6/model_fold5/' | grep -v 'iter1/model_fold1/' | grep -v 'iter9/model_fold6/' | grep -v 'iter10/model_fold1/') ;
    fi

    if [ ! -f lists/finnish/list.finnish.attrib\=${attrib}_mpd\=8_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv ]; then
      python3 shibboleth/scripts/aggregate_sacx.py \
        --docnorm \
        --items_in_list 100 \
        --max_per_doc 8 \
        --selection_threshold $thresh \
        lists/finnish/list.finnish.attrib\=${attrib}_mpd\=8_thresh\=${thresh}_docnorm\=T_tfidf\=F.csv \
        $(find /scratch/project_2006235/shibboleth/models/finnish/ -type f -name ${attrib}_token.jsonl | grep -v 'iter3/model_fold3/') ;
    fi
  done ;
done
