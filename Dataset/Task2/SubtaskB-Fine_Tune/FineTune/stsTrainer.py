import os
import csv
import sys
import math
import gzip
import torch
import random
import numpy as np

from torch.utils.data      import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


sys.path.append( '../preprocessing/sentenceTransformers/' )
sys.path.append( '../preprocessing/LanguageModeling/' )

from createSampleTokenised import _load_csv as load_csv

sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'


def train_model( train_data_file, model_location, outlocation, epochs, seed, save_all, start_from=None ) :

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_batch_size = 4

    train_data_headers, train_data = load_csv( train_data_file, ',' )
    
    train_examples= [
        InputExample( texts=[
            train_data[ i ][ train_data_headers.index( 'sentence1' ) ],
            train_data[ i ][ train_data_headers.index( 'sentence2' ) ]
        ], label=float( train_data[ i ][ train_data_headers.index( 'similarity' ) ] ) ) for i in range( len( train_data ) ) ]
        
    if not start_from is None :
        model_location = os.path.join( outlocation, str( seed ), str( start_from ) )
    else :
        start_from = 0
        
    model = SentenceTransformer( model_location )

    
    train_dataset = SentencesDataset(train_examples, model)

    # train_dataloader = train_examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    train_loss       = losses.CosineSimilarityLoss(model=model)

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

    # import pdb; pdb.set_trace()
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)


    if save_all :

        
        for epoch in range( ( start_from + 1 ), epochs + 1 ) : 
            
            warmup_steps    = math.ceil(len(train_dataloader) * 1  * 0.1) #10% of train data for warm-up
            
            model_save_path = os.path.join( outlocation, str( seed ), str( epoch ) )
            os.makedirs( model_save_path )
        
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=1,
                      evaluation_steps=1000,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path
            )
        
    else : 

            warmup_steps    = math.ceil(len(train_dataloader) * epochs  * 0.1) #10% of train data for warm-up
            
            model_save_path = os.path.join( outlocation, str( seed ), str( epochs ) )
            os.makedirs( model_save_path )
        
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=epochs,
                      evaluation_steps=1000,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path
            )

    

if __name__ == '__main__' :


    if len( sys.argv ) < 3 :
        print( "Require language (EN|PT) and location of tokenized sentence transformer model (e.g. git clone https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransTokenizedNoPreTrain [Use git lfs!])." )
        sys.exit()

    language = sys.argv[1].upper()
    assert language in [ 'EN', 'PT' ]

    
    model_location     = sys.argv[2]
    

    folders = [ 'trainDataAllTokenised', 'trainDataNotTokenised', 'trainDataSelectTokenised' ]

    for folder in folders :

        train_data_file = '../CreateFineTuneData/' + language + '/' + folder + '/sts_train_data.csv'
        
        params = {
            'train_data_file' : train_data_file ,
            'model_location'  : model_location  ,
            'outlocation'     : 'output-no-git/' + language + '/' + folder + '/' ,
            'epochs'          : 4 ,
            'save_all'        : True,
            'start_from'      : None ,
        }


        ## Modify this to continue training from some epoch
        ## params[ 'start_from' ] = 1

    
        if ( not 'start_from' in params.keys() ) or ( params[ 'start_from' ] is None ) : 
            os.makedirs( params[ 'outlocation' ] )

        ## Start from best key of STS trainer so don't have to test multiple seeds here. 
        params[ 'seed' ] = 0
    
        train_model( **params ) 
        
