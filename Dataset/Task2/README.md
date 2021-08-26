
# Task 2: Idiomaticity Representation

Task 2 is tests models' ability to accurately represent sentences regardless of whether or not they contain idiomatic expressions. This is tested using Semantic Text Similarity (STS) and the metric for this task is the Spearman Rank correlation between models' output STS between sentences containing idiomatic expressions and the same sentences with the idiomatic expressions replaced by non-idiomatic paraphrases (which capture the correct meaning of the MWEs). 

We perform all training 5 times with different random seeds and pick the best performing model. 

Please see the paper for more details on the task. 

## Table of Contents

- [Adding Idiom Tokens to 🤗 Transformers Models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#adding-idiom-tokens-to--transformers-models)
- [Creating Sentence Transformers models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#creating-sentence-transformers-models)
- [Creating the Evaluation Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#creating-the-evaluation-data)
- [Generating Pre-Training Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#generating-pre-training-data)
	* [Extract Data from Common Crawl](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#extract-data-from-common-crawl)
	* [Preparing Pre-train data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#preparing-pre-train-data)
- [Subtask A - Pre-Training for Idiom Representation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#subtask-a---pre-training-for-idiom-representation)
	* [Pre-Training](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#pre-training)
	* [Converting to Sentence Transformer Models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#converting-to-sentence-transformer-models)
	* [Evaluation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#evaluation)
- [Subtask B - Fine-Tuning for Idiom Representation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#subtask-b---fine-tuning-for-idiom-representation)
	* [Create Fine-Tuning Data](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#create-fine-tuning-data)
	* [Fine-Tuning](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#fine-tuning)
	* [Evaluation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#evaluation-1)
- [Pre-Trained and Fine-Tuned Models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#pre-trained-and-fine-tuned-models)



## Adding Idiom Tokens to 🤗 Transformers Models

Since we explore the impact of tokenizing MWEs as single tokens (the idiom principle), we first ensure that these tokens are added to pre-trained language models.

This is done using scripts in the [Tokenize folder](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/Tokenize).  

* [downloadModels.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Tokenize/downloadModels.py "downloadModels.py") will download the required model from 🤗 Transformers.
* [updateVocab.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Tokenize/updateVocab.py "updateVocab.py") updates the vocabulary of the model (This uses the "unused" tokens so currently only works for BERT and mBERT. Use tokenizer.add_tokens as described [here](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.add_tokens) for a generic solution. 
*  [tokenCheck.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Tokenize/tokenCheck.py "tokenCheck.py") will run a check to ensure that the tokenizer now tokenizes idioms with a single token. 


## Creating Sentence Transformers models

We use [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) to generate sentence embeddings that can be compared using cosine similarity. 

We modify the original package to allow it to handle the updated tokenization. Please install the version [provided with this repository](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/dependencies/sentence-transformers). 

Here are the steps to create a Sentence Transformer Model: 
* Use [createSentTransformerModel.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/createSentTransformerModel.py "createSentTransformerModel.py") to create a sentence transformer model starting from a model whose tokens have been updated to include idioms (see above). 
* Run [tokenCheck.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/tokenCheck.py "tokenCheck.py") to check that the sentence transformer model uses the new tokens. 
* Use [training_stsbenchmark.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/training_stsbenchmark.py "training_stsbenchmark.py") (and [training_stsbenchmark_PT.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/training_stsbenchmark_PT.py "training_stsbenchmark_PT.py") for Portuguese) to train the model with STS data so it outputs embeddings that can be compared using cosine similarity. 

## Creating the Evaluation Data

Since this task requires models to be self consistent, we need to create evaluation data (or format it for use in our models) using a model that outputs semantic text similarity (such as the one trained above). 

This is done using scripts in the folder [CreateEvaluationData](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/CreateEvaluationData "CreateEvaluationData"). 
* Start with the evaluation data available in the "NoResults" folders for [EN](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/CreateEvaluationData/EN/NoResults/evalData "This path skips through empty directories") and [PT](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/CreateEvaluationData/PT/NoResults/evalData "This path skips through empty directories"). These folders contain additional information regarding tokenization (for select tokenize and all tokenize) and similarities (which is what we need to ensure consistency). This data is created using the script [createEvalData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/createEvalData.py "createEvalData.py"), but it is NOT recommended that you run this script as it might generate a slightly different dataset based on your random number generator.
* Run [predictSentSims.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/predictSentSims.py "predictSentSims.py") (with the STS model created above) to generate sentence similarities. 
* Run [runGlueEval.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/runGlueEval.sh "runGlueEval.sh") with the model used to identify idioms to differentiate between all tokenized and select tokenized (we use the one-shot model from Task 1 A)
* Run [combineCreateFinalEvalData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/combineCreateFinalEvalData.py "combineCreateFinalEvalData.py") to generate the final evaluation data. 

## Generating Pre-Training Data

This step is only required for Subtask A.

The processed pre-training data is available for both [English](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/EN_Pre-Train_Data "EN_Pre-Train_Data") and [Portuguese](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/PT_Pre-Train_Data "PT_Pre-Train_Data"). 

### Extract Data from Common Crawl
This step is only required when not using the pre-training data made available above. 

We obtain pre-train data from the common crawl news corpus. This can be done using scripts in the [ProcessCommonCrawl](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/ProcessCommonCrawl "ProcessCommonCrawl") folder. 
* [processCCNews.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/ProcessCommonCrawl/processCCNews.py "processCCNews.py") will download CC News from 2020 and store relevant files. 
* [createPreTrainData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/ProcessCommonCrawl/createPreTrainData.py "createPreTrainData.py") will create the data required for pre-training with additional files that have information to generate the select tokenize pre-train model (please see paper for details). 

### Preparing Pre-train data 
This step (also) is only required when not using the pre-training data made available above. 

* The output of the previous step should result in the following files: all_replace_data.txt classification_sents.csv no_replace_data.txt vocab_update.txt
* Split all_replace_data.txt into train and eval. (We use split -l 400000 for English and split -l 4000 for PT)
* Run [runGlue.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/runGlue.sh "runGlue.sh") to so as to generate predictions on which usage is idiomatic (This is used to generate the 'select' data)
* Run [createReplaceByPrediction.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/createReplaceByPrediction.py "createReplaceByPrediction.py") to use the predictions above to generate 'select' replaced data for pre-training. 
* Split select_replace_data.txt into train and eval. (We use split -l 400000 for English and split -l 4000 for PT)

## Subtask A - Pre-Training for Idiom Representation

Once the evaluation data and pre-training data have been created and the models have been modified to include single tokens for idioms, these scripts can be used for pre-training and evaluation. 

### Pre-Training 
* Run [preTrain.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/preTrain.py "preTrain.py") to **continue pre-training** from an existing  🤗 Transformers checkpoint (we do not train from scratch). The model used must have tokens associated with MWEs inserted as described in section [Adding Idiom Tokens to 🤗 Transformers Models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#adding-idiom-tokens-to--transformers-models) above. 

### Converting to Sentence Transformer Models

Each of the pre-trained models must be converted to Sentence Transformer models by training them on STS data so their output embeddings can be compared using cosine similarity. This can be done using steps described in the section [Creating Sentence Transformers models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#creating-sentence-transformers-models) above. 

We do this five times with different seeds and pick the model that performs the best on the ordinary STS dataset used to train Sentence Transformers (which does NOT contain any information on the MWEs we work with). 

### Evaluation
You can evaluate the pre-trained representations of MWEs using scripts in the folder [Task2/SubtaskA-Pre_Train/Evaluation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/SubtaskA-Pre_Train/Evaluation "Evaluation"). 
* We test each of the best models from the previous steps using the common script for task 2 evaluation ([task2Evaluation.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Evaluation/task2Evaluation.py "task2Evaluation.py")). 
* You can run all the tests (default model, default model with special MWE tokenization, and models pre-trained with "all" and "select" pre-training data using the script [SubtaskA-Pre_Train/Evaluation/eval.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/Evaluation/eval.sh "eval.sh"). Be sure to update the path of the models. [Please see paper for an explanation of each of these four variations]

## Subtask B - Fine-Tuning for Idiom Representation

Fine-tuning models to better represent idioms also requires creating training data (or formatting training data) in a manner similar to that of creating/formatting evaluation data. This section describes the steps required in formatting the training data, training models and finally the evaluation. 

### Create Fine-Tuning Data

Fine-tuning data can be created using the scripts in the folder [Task2/SubtaskB-Fine_Tune/CreateFineTuneData](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/SubtaskB-Fine_Tune/CreateFineTuneData "CreateFineTuneData"). 
* [createFineTuneData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskB-Fine_Tune/CreateFineTuneData/createFineTuneData.py "createFineTuneData.py") extracts data from the raw json files along with creating files for predicting idiomaticity (required for "all" tokenized and "select" tokenized) and sentences similarity (required for ensuring self consistency). 
* [predictSentSims.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskB-Fine_Tune/CreateFineTuneData/predictSentSims.py "predictSentSims.py") will predict sentence similarity. This script uses a Sentence Transformers model with idiom tokens added (see section Creating Sentence Transformers models and Adding Idiom Tokens to 🤗 Transformers Models). 
* Run [runGlueForTrainData.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskB-Fine_Tune/CreateFineTuneData/runGlueForTrainData.sh "runGlueForTrainData.sh") with the model used to identify idioms to differentiate between "all" tokenized and "select" tokenized (we use the one-shot model from Task 1 A)
* [combineCreateFinalTrainData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskB-Fine_Tune/CreateFineTuneData/combineCreateFinalTrainData.py "combineCreateFinalTrainData.py") combines all the different files and creates the final training data for all three variations (no tokenization change, idioms always replaced with new tokens, idioms replaced by new tokens only when we identify the usage as idiomatic). 

### Fine-Tuning

The data created above can now be used to train model a sentence transformer model. 

**IMPORTANT**: We must start with a model that is already trained on the non-idiomatic STS data as described in the section [Creating Sentence Transformers models](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#creating-sentence-transformers-models) above. The model should be able to handle the special tokens that use for idioms. 

The script [Task2/SubtaskB-Fine_Tune/FineTune/stsTrainer.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskB-Fine_Tune/FineTune/stsTrainer.py "stsTrainer.py") can be used to perform this fine tuning for all variations (no tokenization, "select" tokenization, and "all" tokenization"). 

### Evaluation

The models trained above can be evaluated (all three variations - with no special tokenization, with "all" idioms tokenized, with only those instances of idioms identified to be idiomatic "select" tokenized ) using the same evaluation script: [Task2/Evaluation/task2Evaluation.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Evaluation/task2Evaluation.py "task2Evaluation.py")

The following shell script provides all the required commands: [Task2/SubtaskB-Fine_Tune/Evaluation/evalTask2B.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskB-Fine_Tune/Evaluation/evalTask2B.sh "evalTask2B.sh")

## Pre-Trained and Fine-Tuned Models

The following models associated with Task 2 are publicly available. When training models, we train each 5 times with a different random seed and pick the best performing model (available here). 

**NOTE**: Please note that Sentence Transformer models can't be directly used with the 🤗 Transformers link. They need to be downloaded to local disk (using git clone) before being used. Please remember to use git lfs! 

| No. | 🤗 Transformers Name | Lang | Subtask | Details |
|--|--|--|--|--|
| 1 | [harish/AStitchInLanguageModels-Task2_EN_BERTTokenizedNoPreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_BERTTokenizedNoPreTrain) | EN | A | BERT Base with tokenizer  updated to handle MWEs as single tokens. **No additional pre-training**. |
| 2 | [harish/AStitchInLanguageModels-Task2_EN_BERTTokenizedALLReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_BERTTokenizedALLReplacePreTrain) | EN | A | BERT Base with tokenizer  updated to handle MWEs as single tokens and additionally pre-trained using the "**ALL Replace**" strategy. | 
| 3 | [harish/AStitchInLanguageModels-Task2_EN_BERTTokenizedSelectReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_BERTTokenizedSelectReplacePreTrain) | EN | A | BERT Base with tokenizer  updated to handle MWEs as single tokens and additionally pre-trained using the "**Select Replace**" strategy. |
| 4 | [harish/AStitchInLanguageModels-Task2_EN_SentTransTokenizedNoPreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransTokenizedNoPreTrain) | EN | A | Model No. 1 above converted to Sentence Transformer model with STS training | 
| 5 | [harish/AStitchInLanguageModels-Task2_EN_SentTransALLReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransALLReplacePreTrain) | EN | A | Model No. 2 above converted to Sentence Transformer model with STS training | 
| 6 | [harish/AStitchInLanguageModels-Task2_EN_SentTransSelectReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransSelectReplacePreTrain) | EN | A | Model No. 3 above converted to Sentence Transformer model with STS training |
||||||
| 7 | [harish/AStitchInLanguageModels-Task2_PT_mBERTTokenizedNoPreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_mBERTTokenizedNoPreTrain) | PT | A | Multilingual BERT Base with tokenizer updated to handle MWEs as single tokens. **No additional pre-training**. |
| 8 | [harish/AStitchInLanguageModels-Task2_PT_mBERTTokenizedALLReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_mBERTTokenizedALLReplacePreTrain) | PT | A | Multilingual BERT Base with tokenizer  updated to handle MWEs as single tokens and additionally pre-trained using the "**ALL Replace**" strategy. | 
| 9 | [harish/AStitchInLanguageModels-Task2_PT_mBERTTokenizedSelectReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_mBERTTokenizedSelectReplacePreTrain) | PT | A | Multilingual BERT Base with tokenizer  updated to handle MWEs as single tokens and additionally pre-trained using the "**Select Replace**" strategy. |
| 10 | [harish/AStitchInLanguageModels-Task2_PT_SentTransTokenizedNoPreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_SentTransTokenizedNoPreTrain) | PT | A | Model No. 7 above converted to Sentence Transformer model with (PT) STS training | 
|11| [harish/AStitchInLanguageModels-Task2_PT_SentTransALLReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_SentTransALLReplacePreTrain) | PT | A | Model No. 8 above converted to Sentence Transformer model with (PT) STS training | 
|12| [harish/AStitchInLanguageModels-Task2_PT_SentTransSelectReplacePreTrain](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_SentTransSelectReplacePreTrain) | PT | A | Model No. 9 above converted to Sentence Transformer model with (PT) STS training |
||||||
|13|[harish/AStitchInLanguageModels-Task2_EN_SentTransDefaultFineTuned](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransDefaultFineTuned) | EN | B | Sentence Transformer with **default tokenization** fine tuned on idiomatic STS data | 
|14 | [harish/AStitchInLanguageModels-Task2_EN_SentTransAllTokenizedFineTuned](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransAllTokenizedFineTuned) | EN | B | Sentence Transformer with special idiom tokenization fine tuned on idiomatic STS data tokenized using the "**ALL replace**" strategy. | 
| 15 | [harish/AStitchInLanguageModels-Task2_EN_SentTransSelectTokenizedFineTuned](https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransSelectTokenizedFineTuned) | EN | B | Sentence Transformer with special idiom tokenization fine tuned on idiomatic STS data tokenized using the "**Select replace**" strategy. | 
| 16 | [harish/AStitchInLanguageModels-Task2_PT_SentTransDefaultFineTuned](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_SentTransDefaultFineTuned) | PT | B | Sentence Transformer with **default tokenization** fine tuned on idiomatic (PT) STS data | 
| 17 | [harish/AStitchInLanguageModels-Task2_PT_SentTransAllTokenizedFineTuned](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_SentTransAllTokenizedFineTuned) | PT | B | Sentence Transformer with special idiom tokenization fine tuned on (PT) idiomatic STS data tokenized using the "**ALL replace**" strategy. | 
| 18 | [harish/AStitchInLanguageModels-Task2_PT_SentTransSelectTokenizedFineTuned](https://huggingface.co/harish/AStitchInLanguageModels-Task2_PT_SentTransSelectTokenizedFineTuned) | PT | B | Sentence Transformer with special idiom tokenization fine tuned on (PT) idiomatic STS data tokenized using the "**Select replace**" strategy. | 
