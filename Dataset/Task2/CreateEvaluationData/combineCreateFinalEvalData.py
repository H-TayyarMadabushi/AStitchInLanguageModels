import os
import csv
import sys
import random

from datasets                import load_dataset
from collections             import defaultdict

sys.path.append( '../Utils/' )

from utils                   import tokenise_idiom as idiom_tokeniser
from utils                   import match_idioms
from utils                   import _load_csv as load_csv
 
random.seed( 42 ) 

def _get_sents( location ) :

    headers, data = load_csv( location, delimiter="," )
    correct = list()
    incorrect = list()
    for row in data :
        correct  .append( row[headers.index( 'correct'   ) ] )
        incorrect.append( row[headers.index( 'incorrect' ) ] )
    return correct, incorrect


def _get_sts_data( language, dataset ) :

    location = os.path.join( 'EvalSTSData', language.lower() + '_' + dataset.lower() + '_Eval_STS_Data.csv' )
    first_line, sts_data = load_csv( location, "," )
    sts_data = [ first_line ] + sts_data
    return sts_data

def create_eval_data( sent_location, similatiries, is_not_idiom_info, out_location, tokenise_idiom, select_tokenise, sts_data_to_add ) :
    sent_header, sents               = load_csv( sent_location, "," )
    row_0, sims_data                 = load_csv( similatiries, "," )
    not_idiom_header, not_idiom_data = load_csv( is_not_idiom_info, "\t" )
    sims_data = [ row_0 ] + sims_data

    assert len( sents ) == len( sims_data ) == len( not_idiom_data )

    
    out_header = [ [ 'score', 'sentence1', 'sentence2' ] ]
    out_data   = list()
    
    for index in range( len( sims_data ) ) :
        this_sentence             = sents[ index ][ sent_header.index( 'sent_idiom'              ) ]
        this_sentence_tokenised   = sents[ index ][ sent_header.index( 'sent_idiom_tokenised'    ) ]
        this_other                = sents[ index ][ sent_header.index( 'sent_other'              ) ]
        this_sim                  = sents[ index ][ sent_header.index( 'sim'                     ) ]

        if tokenise_idiom :
            if select_tokenise :
                ## Based on model prediction
                this_pred = not_idiom_data[ index ][ not_idiom_header.index( 'prediction' ) ]
                if int( this_pred ) == 0 : ## 0 is idiomatic
                    this_sentence = this_sentence_tokenised
            else :
                ## Always
                this_sentence = this_sentence_tokenised

        if this_sim == 'None' :
            this_sim =  float( sims_data[ index ][0] )
        else :
            this_sim = float( this_sim )
            assert this_sim == 1.0

        assert this_sentence != this_other 

        out_data.append( [ this_sim, this_sentence, this_other ] )

    out_data += sts_data_to_add
    random.shuffle( out_data )
    out_data = out_header + out_data
    outfile = os.path.join( out_location, dataset + '_final_eval_data.csv' )
    with open( outfile, 'w' ) as csvfile :
        writer = csv.writer( csvfile )
        writer.writerows( out_data )

    print( "Wrote STS evaluation data to {} of size {}: ".format( outfile, len( out_data ) ) ) 

    return 

if __name__ == '__main__' :


    if len( sys.argv ) < 2 :
        print( "Require language as param" )
        sys.exit()

    language = sys.argv[1].upper()
    assert language in [ 'EN', 'PT' ]

    ## Legacy - should always be "True"
    include_sts     = True


    print( "Language: {}".format( language ), flush=True )
        
    
    for out_location, tokenise_idiom, select_tokenise, create_folder in [
            ( 'evalData'               , False, False, False ),
            ( 'evalDataAllTokenised'   , True , False, True  ),
            ( 'evalDataSelectTokenised', True , True , True  ),
    ] :
        if create_folder :
            print( "Created {}".format( out_location ) )
            os.makedirs( out_location )
            
        for dataset in [ 'dev', 'test' ] : 
            params = {
                'sent_location'      : 'evalData/' + dataset + '_rawSimData.csv' ,
                'similatiries'       : 'evalData/' + dataset + '_similatiries.csv'   ,
                'is_not_idiom_info'  : 'evalData/predict_' + dataset + '/predict_results_None.txt' ,
                'out_location'       : out_location   ,
                'tokenise_idiom'     : tokenise_idiom ,
                'select_tokenise'    : select_tokenise,
            }


            sts_data_to_add = _get_sts_data( language, dataset ) 

            if include_sts :
                params[ 'sts_data_to_add' ] = sts_data_to_add
            else : 
                params[ 'sts_data_to_add' ] = list()

            create_eval_data( **params ) 
