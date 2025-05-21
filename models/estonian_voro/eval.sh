module load python-data

python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_raw/loo_type.jsonl -groundtruth ../../data_groundtruth/finnic/test_features.jsonl > loo_eval.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_raw/loo_token.jsonl -groundtruth ../../data_groundtruth/finnic/test_features.jsonl >> loo_eval.txt
# removed model_ev_tok

python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_raw/ig_corr.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl > ig_eval.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_tok/ig_corr.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl >> ig_eval.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_raw/ig_corr_agg_max.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl > ig_eval_agg_max.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_tok/ig_corr_agg_max.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl >> ig_eval_agg_max.txt

python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_raw/shap_corr.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl > shap_eval.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_tok/shap_corr.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl >> shap_eval.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_raw/shap_corr_agg_max.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl > shap_eval_agg_max.txt
python3 ../../scripts/eval_instances_groundtruth.py -predictions model_ev_tok/shap_corr_agg_max.json -groundtruth ../../data_groundtruth/finnic/test_features.jsonl >> shap_eval_agg_max.txt
