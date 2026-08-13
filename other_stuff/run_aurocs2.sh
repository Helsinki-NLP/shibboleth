echo bcms twitter
for attrib in ig shap lime loo; do
  echo -n "$attrib "
  for iter in {1..10}; do
    for fold in {1..10}; do
      echo -n .
      python3 shibboleth/scripts/eval_instances_groundtruth.py \
        -use-whitelist \
        -p /scratch/project_2006235/shibboleth/models/bcms_twitter/iter${iter}/model_fold${fold}/${attrib}_token.jsonl \
        -g /scratch/project_2005047/explainability/data_groundtruth/bcms/twitter/iter_folds/iter${iter}/dev_iter${iter}_fold${fold}_features.jsonl \
        > /scratch/project_2006235/shibboleth/models/bcms_twitter/iter${iter}/model_fold${fold}/${attrib}_results+wl.txt ;
    done;
  done;
  echo ' completed.'
done
