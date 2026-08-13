echo scandinavian
for attrib in ig shap lime loo; do
  echo -n "$attrib "
  for iter in {1..10}; do
    for fold in {1..10}; do
      echo -n .
      python3 shibboleth/scripts/eval_instances_groundtruth.py \
        -p /scratch/project_2006235/shibboleth/models/scandinavian/iter${iter}/model_fold${fold}/${attrib}_token.jsonl \
        -g /scratch/project_2005047/shibboleth/data/scandinavian/iter_folds/iter${iter}/dev_iter${iter}_fold${fold}_features.jsonl \
        > /scratch/project_2006235/shibboleth/models/scandinavian/iter${iter}/model_fold${fold}/${attrib}_results.txt ;
    done;
  done;
  echo ' completed.'
done

echo estonian voro
for attrib in ig shap lime loo; do
  echo -n "$attrib "
  for iter in {1..10}; do
    for fold in {1..10}; do
      echo -n .
      python3 shibboleth/scripts/eval_instances_groundtruth.py \
        -p /scratch/project_2006235/shibboleth/models/estonian_voro/iter${iter}/model_fold${fold}/${attrib}_token.jsonl \
        -g /scratch/project_2005047/shibboleth/data/estonian_voro/iter_folds/iter${iter}/dev_iter${iter}_fold${fold}_features.jsonl \
        > /scratch/project_2006235/shibboleth/models/estonian_voro/iter${iter}/model_fold${fold}/${attrib}_results.txt ;
    done;
  done;
  echo ' completed.'
done

echo bcms setimes
for attrib in ig shap lime loo; do
  echo -n "$attrib "
  for iter in {1..10}; do
    for fold in {1..10}; do
      echo -n .
      python3 shibboleth/scripts/eval_instances_groundtruth.py \
        -p /scratch/project_2006235/shibboleth/models/bcms_setimes/iter${iter}/model_fold${fold}/${attrib}_token.jsonl \
        -g /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/iter_folds/iter${iter}/dev_iter${iter}_fold${fold}_features.jsonl \
        > /scratch/project_2006235/shibboleth/models/bcms_setimes/iter${iter}/model_fold${fold}/${attrib}_results.txt ;
    done;
  done;
  echo ' completed.'
done
