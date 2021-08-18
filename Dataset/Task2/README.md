
# Task 2: Idiomaticity Representation

Task 2 is tests models' ability to accurately represent sentences regardless of whether or not they contain idiomatic expressions. This is tested using Semantic Text Similarity (STS) and the metric for this task is the Spearman Rank correlation between models' output STS between sentences containing idiomatic expressions and the same sentences with the idiomatic expressions replaced by non-idiomatic paraphrases (which capture the correct meaning of the MWEs). 

Please see the paper for more details on the task. 


## Tokenization 

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

This is done using scripts in the folder [Evaluation/CreateEvaluationData](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/Evaluation/CreateEvaluationData). 
* Start with the evaluation data available in the "NoResults" folders for [EN](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/Evaluation/CreateEvaluationData/EN/NoResults/evalData) and [PT](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/Evaluation/CreateEvaluationData/PT/NoResults/evalData "This path skips through empty directories"). These folders contain additional information regarding tokenization (for select tokenize and all tokenize) and similarities (which is what we need to ensure consistency). This data is created using the script [createEvalData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Evaluation/CreateEvaluationData/createEvalData.py "createEvalData.py"), but it is not recomended that you run this script as it might generate a slightly different dataset based on your random number generator.
* Run [predictSentSims.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Evaluation/CreateEvaluationData/predictSentSims.py "predictSentSims.py") (with the STS model created above) to generate sentence similarities. 
* Run [runGlueEval.sh](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Evaluation/CreateEvaluationData/runGlueEval.sh "runGlueEval.sh") with the model used to identify idioms to differentiate between all tokenized and select tokenized (we use the one-shot model from Task 1 A)
* Run [combineCreateFinalEvalData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Evaluation/CreateEvaluationData/combineCreateFinalEvalData.py "combineCreateFinalEvalData.py") to generate the final evaluation data. 

## Setting up Pre-Training Data

We obtain pre-train data from the common crawl news corpus. This can be done using scripts in the [ProcessCommonCrawl](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/Task2/ProcessCommonCrawl "ProcessCommonCrawl") folder. 
* [processCCNews.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/ProcessCommonCrawl/processCCNews.py "processCCNews.py") will download CC News from 2020 and store relevant files. 
* [createPreTrainData.py](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/ProcessCommonCrawl/createPreTrainData.py "createPreTrainData.py") will create the data required for pre-training with additional files that have information to generate the select tokenize pre-train model (please see paper for details).   

