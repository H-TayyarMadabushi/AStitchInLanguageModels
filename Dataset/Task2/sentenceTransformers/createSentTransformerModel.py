"""

Example taken from: https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/examples/training/sts/training_stsbenchmark.py

This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

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

from transformers import AutoTokenizer

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name      = sys.argv[1] 

model_save_path = sys.argv[2] 
seed            = int( sys.argv[3] ) 
if not seed is None : 
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )


word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)


tokenizer      = AutoTokenizer.from_pretrained(
    model_name         , 
    use_fast       = False ,
    max_length     = 510   ,
    force_download = True
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model._first_module().tokenizer = tokenizer



model.save( model_save_path )

print()
print( "Saved model to {}".format( model_save_path ) ) 
print()
print( "Be sure to check using tokenCheck.py" )
print()
