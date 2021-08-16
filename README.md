# AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models

This package contains the dataset AStitchInLanguageModels and associated task information. 

This dataset and associated tasks were introduced in our EMNLP 2021 paper "AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models"

This is a novel dataset consisting of: 
* Naturally occurring sentences (and two surrounding sentences) containing potentially idiomatic MWEs annotated with a fine-grained set of meanings: compositional meaning, idiomatic meaning(s), proper noun and "meta usage". See Tasks ([Task 1](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-1-idiomaticity-detection), [Task 2](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-2-idiomaticity-representation)) for details and Raw Data Section for complete data.
* Paraphrases for each meaning of each MWE; (See [Noun Compound Paraphrase Dataset](#Noun-Compound-Paraphrase-Dataset))

In addition, we use this dataset to define two tasks:
* These tasks are aimed at evaluating i) a model’s ability to detect idiomatic use ([Task 1](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-1-idiomaticity-detection)), and ii) the effectiveness of sentence embeddings in representing idiomaticity ([Task 2](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-2-idiomaticity-representation)).
* These tasks are presented in multilingual, zero-shot, one-shot and few-shot settings.
* We provide strong baselines using state-of-the-art models, including experiments with one-shot and few-shot setups for idiomaticity detection and the use of the ***idiom principle*** for detecting and representing MWEs in contextual embeddings. Our results highlight the significant scope for improvement.

## Table of Contents

## Task 1: Idiomaticity Detection

## Task 2: Idiomaticity Representation

## Noun Compound Paraphrase Dataset
