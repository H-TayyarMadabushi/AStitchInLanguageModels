import os
import sys
import csv

import torch
import logging

from torch.utils.data                 import DataLoader
from sentence_transformers            import SentenceTransformer, LoggingHandler, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers    import STSBenchmarkDataReader

from sklearn.metrics                  import f1_score, accuracy_score
from sklearn.metrics.pairwise         import paired_cosine_distances

sys.path.append( '../Utils/' )

from utils                            import _load_csv as load_csv

def _get_sents( location ) :

    headers, data = load_csv( location, delimiter="," )
    correct = list()
    incorrect = list()
    for row in data :
        correct  .append( row[headers.index( 'sentence1' ) ] )
        incorrect.append( row[headers.index( 'sentence2' ) ] )
    return correct, incorrect

def get_sims( model_location, sentence1, sentence2, dataset, out_location ) : 

    model = SentenceTransformer( model_location )

    sentence1_embeddings = model.encode( sentence1  , batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    sentence2_embeddings = model.encode( sentence2, batch_size=16, show_progress_bar=True, convert_to_numpy=True)


    sims              = 1 - ( paired_cosine_distances( sentence1_embeddings, sentence2_embeddings ) )
    
    outdata   = [ [ sims[ i ] ]  for i in range( len( sims ) ) ]

    outfile = os.path.join( out_location, dataset + '_similatiries.csv' )
    with open( outfile, 'w' ) as csvfile :
        writer = csv.writer( csvfile )
        writer.writerows( outdata )
    print( "Wrote: ", outfile )
    
    
if __name__ == '__main__' :

    ## Will predict similarities of sentence pairs that do NOT have MWEs providing a comparios for evaluation.
    ## This script should be run with a location to a sentence transformer model with updated tokenization.
    ## e.g. git clone https://huggingface.co/harish/AStitchInLanguageModels-Task2_EN_SentTransTokenizedNoPreTrain (Use git lfs!) 
    if len( sys.argv ) < 2 :
        print( "This script should be run with a location to a sentence transformer model with updated tokenization." )
        
    model_location = sys.argv[1]

    for dataset in [ 'dev', 'test' ] : 
        sent1, sent2 = _get_sents( 'evalData/' + dataset + '_dataForSimilarity.csv' )
        
        params = {
            'model_location' : model_location ,
            'dataset'        : dataset      ,
            'sentence1'      : sent1        ,
            'sentence2'      : sent2        ,
            'out_location'   : 'evalData/' ,
        }

        get_sims( **params ) 
