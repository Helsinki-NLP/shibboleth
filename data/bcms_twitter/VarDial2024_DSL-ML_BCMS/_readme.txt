# VarDial 2024 Shared Task on Distinguishing Between Similar Languages - Multiple Labels: Bosnian, Croatian, Montenegrin, Serbian (BCMS)

This archive contains files used in the VarDial 2024 Shared Task on Distinguishing Between Similar Languages - Multiple Labels for the Bosnian - Croatian - Montenegrin - Serbian (BCMS) subtask.

The starting point for this dataset is the one published by Rupnik et al. (2023). It contains geolocated data from the BCMS linguistic area collected from Twitter (rebranded as X).
Each instance contains the full tweet production of a single user, which was manually annotated for the user's country.
The original annotation was single-label, and it was produced by a single annotator. In the version of the data produced here, the test and dev sets were reannotated by multiple annotators, in a multi-label setting. For the details on the reannotation process, please see Miletić and Miletić (2024). We have also excluded retweets from the original data, as these represent reproduced content from a different user account and may not be representative of the language use of the user themselves. 

For details on the shared task, we refer you to Chifu et al. (2024).


## Data
Even though only the dev and test sets were reannotated, this archive also contains the train set for ease of access. 

The data is stored in a two-column TSV format. The first column contains country labels. If multiple labels are present, they are comma-separated and sorted alphabetically. The second column contains the data instance, i.e. the full tweet production of a given user.

The size of each split and the distribution of labels are given below.

Train
bs	45
hr	53
me	34
sr	236
Total	368

Dev
bs	7
bs,hr	4
bs,hr,me,sr	1
bs,me	5
bs,sr	4
hr	16
hr,sr	6
me	4
me,sr	5
sr	70
Total	122

Test
bs	10
bs,hr	3
bs,hr,me	1
bs,me	4
bs,me,sr	2
bs,sr	1
hr	16
hr,sr	2
me	8
me,sr	3
sr	73
Total	123

The reannotation of the test split was done after the publication of Miletić and Miletić (2024) and is therefore not described there. This work was done by three annotators from Serbia. Gold standard annotation was derived using the same procedure as for the dev split.

*NB*: Due to the order in which the files were reannotated, the original test split from the dataset by Rupnik et al. (2023) was used as the dev split in the Shared Task, and the original dev split was used as the test split. 


## License
This dataset is made available under the same conditions as the original dataset by Rupnik et al. (2023).

It is reusable under the Creative Commons Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0, full text here: https://creativecommons.org/licenses/by-sa/4.0/).

## Authors
Data processing, annotation process and annotation processing were conducted by Aleksandra Miletić (Department of Digital Humanities, University of Helsinki) and Filip Miletić (IMS, University of Stuttgart).

## Acknowledgments
The work of Aleksandra Miletić was supported by the CorCoDial project (Academy of Finland, funding decision no. 341798). Filip Miletić was supported by DFG research grant
SCHU 2580/5-1. 

We thank Mikko Aulamo, Yves Scherrer, and Amelie Wührl for their help in setting up the annotation platform.

We are also indebted to our volunteer annotators, whose participation enabled this work: Vesna Arsenović, Katja Bilać, Bojana Damnjanović, Ljubomir Ivanović, Biljana Kaurin, Marijana Kaurin, Maida Kojić McAndrew, Irina Masnikosa, Snežana Naić, Marija Runić, Tibor Weigand, and the remaining 22 participants who wished to remain anonymous.

## Citing
If you reuse this dataset, please cite both the reference to the reannotated dataset distributed here (Miletić and Miletić, 2024) and the reference to the original dataset (Rupnik et al. 2023).

Aleksandra Miletić and Filip Miletić. 2024. A Gold Standard with Silver Linings: Scaling Up Annotation for Distinguishing Bosnian, Croatian, Montenegrin and Serbian. In Proceedings of the Fourth Workshop on Human Evaluation of NLP Systems (HumEval) @ LREC-COLING 2024, pages 36–46, Torino, Italia. ELRA and ICCL.

Peter Rupnik, Taja Kuzman, and Nikola Ljubešić. 2023. BENCHić-lang: A Benchmark for Discriminating between Bosnian, Croatian, Montenegrin and Serbian. In Tenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2023), pages 113–120, Dubrovnik, Croatia. Association for Computational Linguistics.

## References

Adrian-Gabriel Chifu, Goran Glavaš, Radu Tudor Ionescu, Nikola Ljubešić, Aleksandra Miletić, Filip Miletić, Yves Scherrer, and Ivan Vulić. 2024. VarDial Evaluation Campaign 2024: Commonsense Reasoning in Dialects and Multi-Label Similar Language Identification. In Proceedings of the Eleventh Workshop on NLP for Similar Languages, Varieties, and Dialects (VarDial 2024), pages 1–15, Mexico City, Mexico. Association for Computational Linguistics.

Aleksandra Miletić and Filip Miletić. 2024. A Gold Standard with Silver Linings: Scaling Up Annotation for Distinguishing Bosnian, Croatian, Montenegrin and Serbian. In Proceedings of the Fourth Workshop on Human Evaluation of NLP Systems (HumEval) @ LREC-COLING 2024, pages 36–46, Torino, Italia. ELRA and ICCL.

Peter Rupnik, Taja Kuzman, and Nikola Ljubešić. 2023. BENCHić-lang: A Benchmark for Discriminating between Bosnian, Croatian, Montenegrin and Serbian. In Tenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2023), pages 113–120, Dubrovnik, Croatia. Association for Computational Linguistics.
