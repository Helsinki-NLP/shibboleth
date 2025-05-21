#! /bin/bash

if cmp --silent -- filtered_words_msk_sw_sc.txt filtered_words_msk_sw_sc.txt; then
  echo "files contents are identical"
else
  echo "files differ"
fi
