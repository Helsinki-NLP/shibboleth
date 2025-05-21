module load python-data

for BIN in bin1 bin2 bin3 randbin; do
	python3 ../../../scripts/eval_instances_groundtruth.py -predictions "model_$BIN/best/loo_type.jsonl" -groundtruth ../../../data_groundtruth/finnic/test_features.jsonl >> loo_eval.txt
	python3 ../../../scripts/eval_instances_groundtruth.py -predictions "model_$BIN/best/loo_token.jsonl" -groundtruth ../../../data_groundtruth/finnic/test_features.jsonl >> loo_eval.txt
done
# removed model_slide_tok evaluations

# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_raw/ig_corr.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl > ig_eval.txt
# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_tok/ig_corr.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl >> ig_eval.txt
# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_raw/ig_corr_agg_max.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl > ig_eval_agg_max.txt
# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_tok/ig_corr_agg_max.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl >> ig_eval_agg_max.txt

# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_raw/shap_corr.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl > shap_eval.txt
# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_tok/shap_corr.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl >> shap_eval.txt
# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_raw/shap_corr_agg_max.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl > shap_eval_agg_max.txt
# python3 ../../scripts/eval_instances_groundtruth.py -predictions model_slide_tok/shap_corr_agg_max.json -groundtruth ../../data_groundtruth/scandinavian/test_features.jsonl >> shap_eval_agg_max.txt
