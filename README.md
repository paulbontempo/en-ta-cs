# en-ta-cs
## Abstract
This project investigates the relationship between a speaker's sentiment and their language of expression in a code-switching context. Specifically, we create and examine a dataset of mixed English-Tamil text to explore the effect of utterance sentiment on (1) proportion of embedded language to matrix language and (2) frequency of language switches. We perform statistical analysis on this dataset and find that positive utterances show a greater ratio of English to Tamil than negative utterances, and mixed-sentiment utterances show the greatest frequency of language switches when controlling for utterance length. We prove that the emotional content of the message demonstrably influences the language of expression in multilingual settings.

## Intro
Code-switching is the phenomenon of a bilingual speaker changing their language, dialect, or register of expression (Sankoff & Poplack 1981). The normative, unmarked, or privileged language is the matrix language, while the marked or marginalized language is the embedded language. Code-switching can be subcategorized into two types as follows, with the majority of our data comprising the latter.
Inter-discursive code-switching involves less frequent alternation and typically occurs as a response to external pressure or contextual changes, such as the transition from home to work or from one interlocutor to another. 
Switches occur only at discourse or utterance boundaries 
Switches signal speaker identity or match an interlocutor's identity/language competency
Code-mixing (intra-discursive code-switching) is often an unconscious process related to a speaker’s internal attitudes and typically occurs in conversation with other bilingual interlocutors (Muysken 2003). 
Switches can occur at sentence, word, morpheme, or character boundaries
Switches affirm speaker identity or signal speaker attitude towards utterance content
Language status and bilingualism in Southern India
Bilingualism is common in India, with about 12% of the population claiming L2 proficiency in English alongside a native Indian language (Sarma 2025).
In Southern states like Tamil Nadu, English often takes the role of lingua franca (as opposed to Hindi in the North), resulting in academic/professional associations of prestige with English, alongside emotional associations with native languages like Tamil (Eldho & Kumar 2023).
Understanding the motivating factors behind code-switching can help reduce stigma against marginalized languages and improve communication in multilingual settings.

## Hypotheses
**Embedded language proportion hypotheses:**
- H0: Utterance sentiment has no effect on proportion of embedded language to matrix language
- H1: Utterance sentiment predicts the proportion of embedded language to matrix language

**Switch frequency hypotheses:**
- H0: Utterance sentiment has no effect on the language switch frequency
- H1: Utterance sentiment predicts language switch frequency


## Data pipeline (steps 1-4 in Python, steps 5-9 in R): 
1. Preprocess initial data (~44,000 utterances) with sentiment tags from DravidianCodeMix via Chakravarthi et al (2022)
2. Filter any examples in Tamil script to avoid prediction errors on unicode vs latin chars
2. Manual annotation of ~3500 tokens (~500 utterances) with language identification tags
3. Finetune XLM-roberta-base (0.3B params) from Conneau et al (2020) on multiclass prediction task
4. Run inference with fine-tuned model on remaining examples in dataset
5. Generate linear models to determine variable relationships
6. Test for significance
7. Generate complex models with interaction terms with length
7. Model optimization comparison (AnOVa)
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
