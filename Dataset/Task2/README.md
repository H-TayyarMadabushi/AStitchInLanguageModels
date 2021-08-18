
# Task 2: Idiomaticity Representation

Task 2 is tests models' ability to accurately represent sentences regardless of whether or not they contain idiomatic expressions. This is tested using Semantic Text Similarity (STS) and the metric for this task is the Spearman Rank correlation between models' output STS between sentences containing idiomatic expressions and the same sentences with the idiomatic expressions replaced by non-idiomatic paraphrases (which capture the correct meaning of the MWEs). 

Please see the paper for more details on the task. 


## Adding Idiom Tokens to 🤗 Transformers Models

Since we explore the impact of tokenizing MWEs as single tokens (the idiom principle), we first ensure that these tokens are added to pre-trained language models.

This is done using scripts in the [Tokenize folder](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/Tokenize).  

* [downloadModels.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Tokenize/downloadModels.py "downloadModels.py") will download the required model from 🤗 Transformers.
* [updateVocab.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Tokenize/updateVocab.py "updateVocab.py") updates the vocabulary of the model (This uses the "unused" tokens so currently only works for BERT and mBERT. Use tokenizer.add_tokens as described [here](https://github.com/huggingface/tokenizers/issues/507#issuecomment-722275904) for a generic solution. 
*  [tokenCheck.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Tokenize/tokenCheck.py "tokenCheck.py") will run a check to ensure that the tokenizer now tokenizes idioms with a single token. 

## Creating Sentence Transformers models

We use [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) to generate sentence embeddings that can be compared using cosine similarity. 

We modify the original package to allow it to handle the updated tokenization. Please install the version [provided with this repository](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/dependencies/sentence-transformers). 

Here are the steps to create a Sentence Transformer Model: 
* Use [createSentTransformerModel.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/createSentTransformerModel.py "createSentTransformerModel.py") to create a sentence transformer model starting from a model whose tokens have been updated to include idioms (see above). 
* Run [tokenCheck.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/tokenCheck.py "tokenCheck.py") to check that the sentence transformer model uses the new tokens. 
* Use [training_stsbenchmark.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/training_stsbenchmark.py "training_stsbenchmark.py") ( and [training_stsbenchmark_PT.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/sentenceTransformers/training_stsbenchmark_PT.py "training_stsbenchmark_PT.py") for Portuguese) to train the model with STS data so it outputs embeddings that can be compared using cosine similarity. 

## Creating the Evaluation Data

Since this task requires models to be self consistent, we need to create evaluation data using a model that outputs semantic text similarity (such as the one trained above). 

This is done using scripts in the folder [CreateEvaluationData](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/CreateEvaluationData "CreateEvaluationData"). 
* Start with the evaluation data available in the "NoResults" folders for [EN](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/CreateEvaluationData/EN/NoResults/evalData "This path skips through empty directories") and [PT](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/CreateEvaluationData/PT/NoResults/evalData "This path skips through empty directories"). These folders contain additional information regarding tokenization (for select tokenize and all tokenize) and similarities (which is what we need to ensure consistency). This data is created using the script [createEvalData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/createEvalData.py "createEvalData.py"), but it is NOT recommended that you run this script as it might generate a slightly different dataset based on your random number generator.
* Run [predictSentSims.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/predictSentSims.py "predictSentSims.py") (with the STS model created above) to generate sentence similarities. 
* Run [runGlueEval.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/runGlueEval.sh "runGlueEval.sh") with the model used to identify idioms to differentiate between all tokenized and select tokenized (we use the one-shot model from Task 1 A)
* Run [combineCreateFinalEvalData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/CreateEvaluationData/combineCreateFinalEvalData.py "combineCreateFinalEvalData.py") to generate the final evaluation data. 

## Generating Pre-Training Data

This step is only required for Subtask A.

The processed pre-training data is available for both [English](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/EN_Pre-Train_Data "EN_Pre-Train_Data") and [Portuguese](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/PT_Pre-Train_Data "PT_Pre-Train_Data"). 

### Extract Data from Common Crawl
This step is only required when not using the pre-training data made avilable above. 

We obtain pre-train data from the common crawl news corpus. This can be done using scripts in the [ProcessCommonCrawl](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/ProcessCommonCrawl "ProcessCommonCrawl") folder. 
* [processCCNews.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/ProcessCommonCrawl/processCCNews.py "processCCNews.py") will download CC News from 2020 and store relevant files. 
* [createPreTrainData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/ProcessCommonCrawl/createPreTrainData.py "createPreTrainData.py") will create the data required for pre-training with additional files that have information to generate the select tokenize pre-train model (please see paper for details). 

### Preparing Pre-train data 
This step also is only required when not using the pre-training data made avilable above. 

* The output of the previous step should result in the following files: all_replace_data.txt classification_sents.csv no_replace_data.txt vocab_update.txt
* Split all_replace_data.txt into train and eval. (We use split -l 400000 for English and split -l 4000 for PT)
* Run [runGlue.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/runGlue.sh "runGlue.sh") to so as to generate predictions on which usage is idiomatic (This is used to generate the 'select' data)
* Run [createReplaceByPrediction.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/createReplaceByPrediction.py "createReplaceByPrediction.py") to use the predictions above to generate 'select' replaced data for pre-training. 

## Subtask A - Pre-Training for Idiom Representation

Once the evaluation data and pre-training data have been created and the models have been modified to include single tokens for idioms, these scripts can be used for pre-training and evaluation. 

### Pre-Training 
* Run [preTrain.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/preTrain.py "preTrain.py") to continue pre-training from an existing  🤗 Transformers checkpoint. The model used must have tokens associated with MWEs inserted as described in section "Adding Idiom Tokens to 🤗 Transformers Models" above. 

### Converting to Sentence Transformer Models

Each of the pre-trained models must be converted to Sentence Transformer models by training them on STS data so their output embeddings can be compared using cosine similarity. This can be done using steps described in the section "Creating Sentence Transformers models" above. 

We do this five times with different seeds and pick the model that performs the best on the ordinary STS dataset used to train Sentence Transformers (which does NOT contain any information on the MWEs we work with). 

### Evaluation
You can evaluate the pre-trained representations of MWEs using scripts in the folder [Task2/SubtaskA-Pre_Train/Evaluation](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/SubtaskA-Pre_Train/Evaluation "Evaluation"). 
* We each of the best models from the previous steps using the script [ask2SubtaskAEvaluation.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/Evaluation/task2SubtaskAEvaluation.py "task2SubtaskAEvaluation.py")
* You can run all the tests (default model, default model with special MWE tokenization, and models pre-trained with "all" and "select" pre-training data using the script [eval.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/Evaluation/eval.sh "eval.sh"). [Please see paper for explanation on each of these four models]

## Subtask B - Fine-Tuning for Idiom Representation

### Create Fine-Tuning Data

### Fine-Tuning

### Evaluation
