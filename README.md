
# AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models

This package contains the dataset AStitchInLanguageModels and associated task information. 

This dataset and associated tasks were introduced in our (findings of) EMNLP 2021 paper "[AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models](https://aclanthology.org/2021.findings-emnlp.294/)". Please [cite](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#citation) this paper if you use any of the data or methods from this package. 

**The paper will be made available by the 9th of September**

This is a novel dataset consisting of: 
* Naturally occurring sentences (and two surrounding sentences) containing potentially idiomatic MWEs annotated with a fine-grained set of meanings: compositional meaning, idiomatic meaning(s), proper noun and "meta usage". See Tasks ([Task 1](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-1-idiomaticity-detection), [Task 2](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-2-idiomaticity-representation)) for details and Raw Data Section for complete data.
* Data in both Portuguese and English
* Paraphrases for each meaning of each MWE; (See [Extended Noun Compound Senses Dataset](#Extended-Noun-Compound-Senses-Dataset))

In addition, we use this dataset to define two tasks:
* These tasks are aimed at evaluating i) a model’s ability to detect idiomatic use ([Task 1](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-1-idiomaticity-detection)), and ii) the effectiveness of sentence embeddings in representing idiomaticity ([Task 2](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-2-idiomaticity-representation)).
* These tasks are presented in multilingual, zero-shot, one-shot and few-shot settings.
* We provide strong baselines using state-of-the-art models, including experiments with one-shot and few-shot setups for idiomaticity detection and the use of the ***idiom principle*** for detecting and representing MWEs in contextual embeddings. Our results highlight the significant scope for improvement.

## Table of Contents

* [Prerequisites](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#prerequisites)
* [Task 1: Idiomaticity Detection](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#task-1-idiomaticity-detection)
* [Task 2: Idiomaticity Representation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#task-2-idiomaticity-representation)
* [Extended Noun Compound Senses Dataset](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#extended-noun-compound-senses-dataset)
* [Task Independent Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#task-independent-data)
* [Citation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/README.md#citation)

## Prerequisites 

The scripts in this package have been tested using Python 3.8.6 and PyTorch 1.7.1. The additionally require the following packages. Please note that this will overwrite existing versions of your package. For this reason we suggest you use a [virtual environment](https://docs.python.org/3/library/venv.html).

### Install the location version of Sentence Transformers
While we use [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) to generate sentence embeddings that can be compared using cosine similarity, we make some changes to ensure that it can use custom tokenizers. Please install the location version available at [dependencies/sentence-transformers](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/dependencies/sentence-transformers "This path skips through empty directories"). 

```bash 
cd AStitchInLanguageModels/dependencies/sentence-transformers
pip3 install -e . 
```

### Editable version of 🤗 Transformers (4.7.0.dev0)

Download version 4.7.0 from [here](https://github.com/huggingface/transformers/releases/tag/v4.7.0).
```bash
cd transformers-4.7.0
pip3 install -e . 
```


### Other dependencies

```bash
pip3 install datasets==1.6.1
pip3 install tqdm==4.49.0
pip3 install nltk==3.6.2
```
And from your Python prompt:
```python
>>> import nltk
>>> nltk.download('punkt')
```


## Task 1: Idiomaticity Detection

The first task we propose is designed to evaluate the extent to which models can identify idiomaticity in text and consists of two Subtasks: a _coarse-grained_ classification task (Subtask A) and a _fine-grained_ classification task (Subtask B). The evaluation metric for this task is F1. 

The data associated with this Task can be found in [this folder](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task1). Data is split into zero-shot, one-shot and few-shot data in both Portuguese and English. Please see the paper for a detailed description of the task and methods. 

We used 🤗 Transformers ([this script](https://github.com/huggingface/transformers/blob/62ba3b6b43975e759851336b566852252be00669/examples/pytorch/text-classification/run_glue.py), local copy with F1 evaluation [available here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Utils/run_glue_f1_macro.py)) for training with the following hyperparameters. Further details are available in the paper. 


```bash
    python run_glue.py \
    	--model_name_or_path $model \
    	--do_train \
    	--do_eval \
    	--max_seq_length 128 \
    	--per_device_train_batch_size 32 \
    	--learning_rate 2e-5 \
    	--num_train_epochs 9 \
    	--evaluation_strategy "epoch" \
    	--output_dir $output_dir \
    	--seed $seed \
    	--train_file      $train_file \
    	--validation_file $dev_file \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch"  \
        --load_best_model_at_end \
        --metric_for_best_model "f1" \
        --save_total_limit 3
```



## Task 2: Idiomaticity Representation

Task 2 is the more challenging task of creating sentence embeddings that accurately represent sentences regardless of whether or not they contain idiomatic expressions. This is tested using Semantic Text Similarity (STS) and the metric for this task is the Spearman Rank correlation between models' output STS between sentences containing idiomatic expressions and the same sentences with the idiomatic expressions replaced by non-idiomatic paraphrases (which capture the correct meaning of the MWEs). Please see the paper for more details on the task. 

Complete details of this task including the data and models is available in the task folder: [AStitchInLanguageModels/Dataset/Task2/](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2). This includes details on the following: 

-   [Adding Idiom Tokens to  🤗  Transformers Models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#adding-idiom-tokens-to--transformers-models)
-   [Creating Sentence Transformers models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#creating-sentence-transformers-models)
-   [Creating the Evaluation Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#creating-the-evaluation-data)
-   [Generating Pre-Training Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#generating-pre-training-data)
    -   [Extract Data from Common Crawl](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#extract-data-from-common-crawl)
    -   [Preparing Pre-train data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#preparing-pre-train-data)
-   [Task 2 Subtask A - Pre-Training for Idiom Representation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#subtask-a---pre-training-for-idiom-representation)
    -   [Pre-Training](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#pre-training)
    -   [Converting to Sentence Transformer Models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#converting-to-sentence-transformer-models)
    -   [Evaluation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#evaluation)
-   [Task 2 Subtask B - Fine-Tuning for Idiom Representation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#subtask-b---fine-tuning-for-idiom-representation)
    -   [Create Fine-Tuning Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#create-fine-tuning-data)
    -   [Fine-Tuning](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#fine-tuning)
    -   [Evaluation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#evaluation-1)
-   [Pre-Trained and Fine-Tuned Models for Task 2](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#pre-trained-and-fine-tuned-models)



## Extended Noun Compound Senses Dataset

We also provide an [Extended Noun Compound Senses dataset](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Extended_Noun_Compound_Senses_Dataset) (ExNC dataset) that is highly granular. This data differs from previous sense datasets in that: 
 * it provides all possible senses,
 * we ensure that meanings provided are as close to the original phrase as possible to ensure that this dataset is an adversarial dataset, 
 * we highlight purely compositional noun compounds. 

Please see the [associated data folder](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Extended_Noun_Compound_Senses_Dataset) for more details. 

## Task Independent Data

You can download the Task independent annotated data from [this folder](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/TaskIndependentData). The data format is described in the README available in the same folder. 

Where possible, please use the training, development and test splits provided so results can remain comparable.

## Citation

If you make use of this work, please cite us: 

```bibtex
@inproceedings{tayyar-madabushi-etal-2021-astitchinlanguagemodels-dataset,
    title = "{AS}titch{I}n{L}anguage{M}odels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models",
    author = "Tayyar Madabushi, Harish  and
      Gow-Smith, Edward  and
      Scarton, Carolina  and
      Villavicencio, Aline",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.294",
    pages = "3464--3477",
}```
