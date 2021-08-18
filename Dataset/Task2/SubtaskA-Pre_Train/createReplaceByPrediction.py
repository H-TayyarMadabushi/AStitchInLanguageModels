import os
import csv
import sys

from tqdm        import tqdm
from collections import defaultdict

sys.path.append( '../preprocessing/LanguageModeling/' )

from tokeniseIdiom    import tokenise_idiom
from createForDevTest import match_idioms


def _load_csv( location, delimiter ) :

    if delimiter is None :
        with open( location ) as infile :
            data = infile.read().lstrip().rstrip().split( '\n' )
            header = data[0]
            data   = data[1:]
        return header, data
            
    
    header = None
    data   = list()
    with open( location ) as csvfile :
        reader = csv.reader( csvfile, delimiter=delimiter )
        for row in reader :
            if header is None :
                header = row
                continue
            data.append( row )
    return header, data        
    

def create_merge_data( sents_location, preds_location, data_location, out_location ) :
    sents_header, sents_data = _load_csv( sents_location, ','  )
    preds_header, preds_data = _load_csv( preds_location, '\t' )

    assert len( preds_data ) == len( sents_data )
    sents_with_preds = list()
    for index in range( len( preds_data ) ) :
        sents_with_preds.append( [
            sents_data[ index ][ sents_header.index( 'sentence1'  ) ],
            sents_data[ index ][ sents_header.index( 'sentence2'  ) ],
            preds_data[ index ][ preds_header.index( 'prediction' ) ],
        ] )

    sents_prediction_dict = defaultdict( list )
    for sent in sents_with_preds :
        sents_prediction_dict[ sent[0] ] = sent[1:]
        
    _, sentences = _load_csv( data_location, None )
    updated_sentences = [ 'text' ]
    found_sents = list()
    replaced    = 0
    kept        = 0
    for index in tqdm( range( len( sentences ) ) ) :
        sent = sentences[ index ]
        if sent in sents_prediction_dict.keys() : 
            found_sents.append( sent )
            idiom, prediction = sents_prediction_dict[ sent ] ## 1 is literal prediction
            idiom   = idiom.lower()
            idiom_words = defaultdict( list )
            idiom_words[ idiom.split()[0] ] =  [ idiom.split()[1] ]
            matched = match_idioms( idiom_words, sent )
            if int( prediction ) == 0 and len( matched ) > 0 : 
                replaced += 1
                for single_match in matched :
                    sent = sent.replace( single_match, tokenise_idiom( single_match ) )
            else : 
                kept += 1
        updated_sentences.append( sent )
        assert ( index + 2 ) == len( updated_sentences )

    assert set( sents_prediction_dict.keys() ) - set( found_sents ) == set()
    assert ( len( sentences ) + 1 ) == len( updated_sentences )  ## Header

    print( "Replaced, kept:", replaced, kept ) 

    with open( out_location, 'w' ) as outfile :
        for row in updated_sentences :
            outfile.write( row + '\n' )
    print( "Wrote: ", out_location )
    
    

if __name__ == '__main__' :


    if len( sys.argv ) < 2 :
        print( "Require Language (EN|PT)" )
        sys.exit()

    language = sys.argv[1]
    assert language.upper() in ['EN', 'PT']
        
    datapath = '../' + language.upper() + '_Pre-Train_Data/'
    params = {
        'sents_location' : datapath + 'classification_sents.csv'          , 
        'data_location'  : datapath + 'no_replace_data.txt'               ,
        'preds_location' : datapath + 'predict/predict_results_None.txt'  ,
        'out_location'   : datapath + 'select_replace_data.txt'           ,
    }
    
    create_merge_data( **params )
