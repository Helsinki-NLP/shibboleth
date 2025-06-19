#! /bin/sh

module load python-data

#python3 filter_sort_explanations.py model_3cl_db/explanations.json 100 > model_3cl_db/filtered_words.txt
#python3 filter_sort_explanations.py model_4cl_db/explanations.json 100 > model_4cl_db/filtered_words.txt
#python3 filter_sort_explanations.py model_5cl_db/explanations.json 100 > model_5cl_db/filtered_words.txt

#python3 filter_sort_explanations.py model_3cl_xlmr/explanations.json 100 > model_3cl_xlmr/filtered_words.txt
#python3 filter_sort_explanations.py model_4cl_xlmr_2/explanations.json 100 > model_4cl_xlmr_2/filtered_words.txt
#python3 filter_sort_explanations.py model_5cl_xlmr/explanations.json 100 > model_5cl_xlmr/filtered_words.txt

python3 filter_sort_explanations.py model_4cl_gb/explanations.json 100 > model_4cl_gb/filtered_words.txt
