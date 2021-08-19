"""

Example taken from: https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/examples/training/sts/training_stsbenchmark.py

This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""

from torch.utils.data import DataLoader
import math
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

sys.stdout.flush()
sys.stderr.flush()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)



#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base



is_st_model = True

if len( sys.argv ) < 3 :
    raise Exception( "Require path of model to test and dataset location" ) 

model_name       = sys.argv[1] 
dataset_location = sys.argv[2] 
seed             = 42
if not seed is None : 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if not is_st_model :
    logging.info( "Not ST model, building ... " )
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else :
    logging.info( "Loading ST model ... " )
    model = SentenceTransformer( model_name )
    
# Convert the dataset to a DataLoader ready for training
logging.info("Reading dataset")




# Read the dataset
train_batch_size = 4
num_epochs       = 4


for dataset in [ 'dev', 'test' ] :
    test_samples = list()
    with open( os.path.join( dataset_location, dataset + '_final_eval_data.csv' ) ) as csvfile :
        reader = csv.DictReader( csvfile )
        for row in reader :
            score = float(row['score']) 
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
            test_samples.append( inp_example )
            
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='Evaluation on '+dataset+'')
    test_evaluator(model)
