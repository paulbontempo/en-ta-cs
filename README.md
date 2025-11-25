# en-ta-cs
Project investigating the relationship between the proportion of two languages in an utterance and the sentiment of that utterance. Explores this question through a dataset of English and Tamil bilingual codeswitching, scraped from YouTube comments left primarily by native Tamil speakers who are also bilingual or L2 users of English. 

Code-switching is the phenomenon of a bilingual speaker changing their language, dialect, or register of expression. It can occur both interdiscursively and intradiscursively; in the former case, a speaker may switch to signal identity and to match their interlocutor's identity or language competency, while in the latter case, a speaker assumes that their interlocutor shares competency in both languages, and thus switches to affirm identity and to signal attitudes towards the content of their speech. Within the category of intradiscursive code-switching (also called code-mixing), speakers may switch at utterance boundaries, word boundaries, or morpheme boundaries. 

This project investigates the relationship between a speaker's sentiment and their language of expression within a code-switching context. Specifically, we examine a dataset of English-Tamil code-switching scraped from YouTube comments on Indian movie trailers. We propose the alternative hypothesis that positive sentiment utterances are more likely to contain a lower proportion of English words than mixed-sentiment utterances, because in the sociolinguistic context of Southern India, English is commonly a prestige language associated with academic and professional contexts, while Tamil is more often associated with personal and emotional expression. Additionally, we investigate the relationship between an utterance's sentiment and its frequency of switches, with the alternative hypothesis that negative sentiment predicts more frequent changes to language of expression in longer utterances, while shorter utterances are unlikely to show a significant relationship between sentiment and switch frequency. 

Data pipeline (steps 1-4 in Python, steps 5-9 in R): 
1. Preprocess initial data (~44,000 utterances) with sentiment tags from DravidianCodeMix via Chakravarthi et al (2022)
2. Filter any examples in Tamil script to avoid prediction errors on unicode vs latin chars
2. Manual annotation of ~3500 tokens (~500 utterances) with language identification tags
3. Finetune XLM-roberta-base (0.3B params) from Conneau et al (2019) on multiclass prediction task
4. Run inference with fine-tuned model on remaining utterances in dataset
5. Generate linear models to determine variable relationships
6. Test for significance
7. Model optimization comparison
8. Test for assumptions (heteroscedasticity, multicollinearity, normal distribution)
9. Plot results

Citations: 

DravidianCodeMix base dataset:
```
@article{Chakravarthi2022,
author={Chakravarthi, Bharathi Raja
and Priyadharshini, Ruba
and Muralidaran, Vigneshwaran
and Jose, Navya
and Suryawanshi, Shardul
and Sherly, Elizabeth
and McCrae, John P.},
title={DravidianCodeMix: sentiment analysis and offensive language identification dataset for Dravidian languages in code-mixed text},
journal={Language Resources and Evaluation},
year={2022},
month={Feb},
day={04},
abstract={This paper describes the development of a multilingual, manually annotated dataset for three under-resourced Dravidian languages generated from social media comments. The dataset was annotated for sentiment analysis and offensive language identification for a total of more than 60,000 YouTube comments. The dataset consists of around 44,000 comments in Tamil-English, around 7000 comments in Kannada-English, and around 20,000 comments in Malayalam-English. The data was manually annotated by volunteer annotators and has a high inter-annotator agreement in Krippendorff's alpha. The dataset contains all types of code-mixing phenomena since it comprises user-generated content from a multilingual country. We also present baseline experiments to establish benchmarks on the dataset using machine learning and deep learning methods. The dataset is available on Github and Zenodo.},
issn={1574-0218},
doi={10.1007/s10579-022-09583-7},
url={https://doi.org/10.1007/s10579-022-09583-7}
}
```

xlm-roberta-base: 
```
@article{DBLP:journals/corr/abs-1911-02116,
  author    = {Alexis Conneau and
               Kartikay Khandelwal and
               Naman Goyal and
               Vishrav Chaudhary and
               Guillaume Wenzek and
               Francisco Guzm{\'{a}}n and
               Edouard Grave and
               Myle Ott and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  journal   = {CoRR},
  volume    = {abs/1911.02116},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.02116},
  eprinttype = {arXiv},
  eprint    = {1911.02116},
  timestamp = {Mon, 11 Nov 2019 18:38:09 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1911-02116.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
