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



## WARNING: Convert using createSentTransformerModel.py to ensure tokenization works.
is_st_model = True
model_name      = sys.argv[1] if len( sys.argv ) > 1 else None
if model_name is None :
    raise Exception( "Must provide model" )

model_save_path = sys.argv[2] if len( sys.argv ) > 2 else None
if model_save_path is None : 
    model_save_path = 'output-no-git/v3/'

seed            = int( sys.argv[3] ) if len( sys.argv ) > 3 else None
if not seed is None : 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

train           = int( sys.argv[4] ) if len( sys.argv ) > 4 else 1
if train == 1 :
    train = True
else :
    train = False
    logging.info("Will only evaluate")





model = None
if not is_st_model :
    print( "Not ST model, building ... ", flush=True )
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else :
    print( "Loading ST model ... ", flush=True )
    model = SentenceTransformer( model_name )
    
# Convert the dataset to a DataLoader ready for training
logging.info("Read assin2 dataset")


# Read the dataset
train_batch_size = 4
num_epochs       = 4

train_samples    = list()
dev_samples      = list()
test_samples     = list()


from datasets import load_dataset
for split in [ 'train', 'validation', 'test' ] :
    dataset = load_dataset( 'assin2', split=split )
    for elem in dataset :
        ## {'entailment_judgment': 1, 'hypothesis': 'Uma criança está segurando uma pistola de água', 'premise': 'Uma criança risonha está segurando uma pistola de água e sendo espirrada com água', 'relatedness_score': 4.5, 'sentence_pair_id': 1}
        score = float( elem['relatedness_score'] ) / 5.0 # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[elem['hypothesis'], elem['premise']], label=score)
        if split == 'validation':
            dev_samples.append(inp_example)
        elif split == 'test':
            test_samples.append(inp_example)
        elif split == 'train' :
            train_samples.append(inp_example)
        else :
            raise Exception( "Unknown split. Should be one of ['train', 'test', 'validation']." )


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
if train : 
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
    )


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
