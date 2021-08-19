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

sys.path.append( '../../LanguageModeling/' )
sys.path.append( '../' )

from createSampleTokenised            import _load_csv as load_csv

def _get_sents( location ) :

    headers, data = load_csv( location, delimiter="," )
    correct = list()
    incorrect = list()
    for row in data :
        correct  .append( row[headers.index( 'correct'   ) ] )
        incorrect.append( row[headers.index( 'incorrect' ) ] )
    return correct, incorrect

def get_sims( model_location, correct, incorrect, out_location ) : 

    model = SentenceTransformer( model_location )

    correct_embeddings   = model.encode( correct  , batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    incorrect_embeddings = model.encode( incorrect, batch_size=16, show_progress_bar=True, convert_to_numpy=True)


    incorrect_correct = 1 - (paired_cosine_distances(incorrect_embeddings, correct_embeddings))
    correct_incorrect = 1 - (paired_cosine_distances(correct_embeddings, incorrect_embeddings))
    
    outdata   = [ [ incorrect_correct[ i ], correct_incorrect[ i ] ] for i in range( len( incorrect_correct ) ) ]

    outfile = os.path.join( out_location, 'similatiries.csv' )
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
        
    model_location     = sys.argv[1]
    correct, incorrect = _get_sents( 'trainData/trainToPredict.csv' )

    
    params = {
        'model_location' : model_location ,
        'correct'        : correct   ,
        'incorrect'      : incorrect ,
        'out_location'   : 'trainData/' ,
    }

    get_sims( **params ) 
