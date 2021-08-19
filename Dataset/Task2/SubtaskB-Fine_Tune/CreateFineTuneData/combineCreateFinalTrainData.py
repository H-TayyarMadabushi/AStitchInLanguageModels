import os
import csv
import sys

from collections  import defaultdict

sys.path.append( '../../Utils/' )

from utils        import tokenise_idiom as idiom_tokeniser
from utils        import match_idioms, create_idiom_word_dict
from utils        import _load_csv as load_csv

def _get_sents( location ) :

    headers, data = load_csv( location, delimiter="," )
    correct = list()
    incorrect = list()
    for row in data :
        correct  .append( row[headers.index( 'correct'   ) ] )
        incorrect.append( row[headers.index( 'incorrect' ) ] )
    return correct, incorrect


def create_train_data( sent_location, similatiries, is_not_idiom_info, out_location, tokenise_strategy ) :
    sent_header, sents               = load_csv( sent_location, "," )
    row_0, sims_data                 = load_csv( similatiries, "," )
    not_idiom_header, not_idiom_data = load_csv( is_not_idiom_info, "\t" )

    sims_data = [ row_0 ] + sims_data

    assert len( sents ) == len( sims_data ) == len( not_idiom_data )

    train_data = [ [ 'sentence1', 'sentence2', 'similarity' ] ]
    for index in range( len( sims_data ) ) :
        this_sentence  = sents[ index ][ sent_header.index( 'original'   ) ]
        this_correct   = sents[ index ][ sent_header.index( 'correct'    ) ]
        this_incorrect = sents[ index ][ sent_header.index( 'incorrect'  ) ]
        this_idiom     = sents[ index ][ sent_header.index( 'idiom'      ) ]
        this_sim       = sims_data[ index ][0]
        this_pred      = not_idiom_data[ index ][ not_idiom_header.index( 'prediction' ) ]

        assert tokenise_strategy.lower() in [ 'select', 'all', 'none' ]
        if ( int( this_pred ) == 0 and tokenise_strategy.lower() == 'select' ) or ( tokenise_strategy.lower() == 'all' ) : ## 0 is idiomatic
    
            idiom_word_dict = create_idiom_word_dict( [ this_idiom ] )
    
            matched_idioms = match_idioms( idiom_word_dict, this_sentence )
    
            if len( matched_idioms ) == 0 :
                print( "NO IDIOM!"  )
                print( this_idiom )
                print( this_sentence )
                # import pdb; pdb.set_trace()
                # matched_idioms = match_idioms( idiom_word_dict, this_sentence )
                continue
    
            this_sentence = this_sentence.replace( this_idiom, idiom_tokeniser( this_idiom ) )

            
        positive_example = [ this_sentence, this_correct, 1          ]
        negative_example = [ this_sentence, this_incorrect, this_sim ]

        train_data.append( positive_example )
        train_data.append( negative_example )

    outfile = os.path.join( out_location, 'sts_train_data.csv' )
    with open( outfile, 'w' ) as csvfile :
        writer = csv.writer( csvfile )
        writer.writerows( train_data )

    print( "Wrote STS train data to: ", outfile ) 

    return 

if __name__ == '__main__' :


    for tokenise_strategy, out_location in [
            ( 'none'  , 'trainDataNotTokenised' ),
            ( 'all'   , 'trainDataAllTokenised' ),
            ( 'select', 'trainDataSelectTokenised' )
    ] :
        params = {
            'sent_location'      : 'trainData/trainToPredict.csv' ,
            'similatiries'       : 'trainData/similatiries.csv'   ,
            'is_not_idiom_info'  : 'trainData/predictions/predict_results_None.txt' ,
            'out_location'       : out_location      ,
            'tokenise_strategy'  : tokenise_strategy ,
        }

        os.makedirs( out_location )

        create_train_data( **params ) 

