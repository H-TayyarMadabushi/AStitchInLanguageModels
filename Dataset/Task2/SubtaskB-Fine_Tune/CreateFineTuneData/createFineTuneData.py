import os
import re
import sys
import csv
import copy
import json
import time
import random

from pathlib     import Path
from collections import defaultdict

sys.path.append( '../../Utils/' )

from utils       import tokenise_idiom as idiom_tokeniser
from utils       import match_idioms, create_idiom_word_dict

def create_data( json_location, out_location, datasets, force_idiomatic, tokenise_idiom ) :

    assert not force_idiomatic 

    with open( json_location, encoding='utf-8') as fh:
        all_data = json.load(fh)

    data = list()
    for dataset in datasets : 
        data += all_data[ dataset ]
        

    original    = list()
    correct     = list()
    incorrect   = list()
    sent_idioms = list() # Idioms per sent (example)
    idioms      = list() # All idioms
    for idiom in data :
        
        this_idiom = idiom[0][1]
        idioms.append( this_idiom.lower().lstrip().rstrip() )
        
        meanings   = idiom[0][2:8]
        del_meanings = [ 'None', 'Proper Noun', 'Meta Usage' ] 
        new_meanings = list()
        for meaning in meanings : 
            if meaning in del_meanings : 
                continue
            new_meanings.append( meaning ) 
        meanings = new_meanings

        for sent in idiom :
            label = sent[9] 
            if not label in meanings : 
                continue
            if force_idiomatic and sent[8] == 1 : 
                continue
            this_sentence  = sent[11]
            
            not_label      = copy.copy( meanings )
            not_label.remove( label )

            if len( not_label ) == 0 :
                continue

            ### Should this be flags=re.I below ?
            not_label      = random.choice( not_label                     )
            this_correct   = re.sub( this_idiom, label, this_sentence     )
            this_incorrect = re.sub( this_idiom, not_label, this_sentence )

            idiom_phrase = sent[1]
            idiom_word_dict = create_idiom_word_dict( [ idiom_phrase ] ) 

            matched_idioms = match_idioms( idiom_word_dict, this_sentence )

            if len( matched_idioms ) == 0 :
                print( "WARNING: Can't identify idiom {} in sentence {} (possibly because of tokenisation!)".format( idiom_phrase, this_sentence ) )
                print()
                    # import pdb; pdb.set_trace()
                    # matched_idioms = match_idioms( idiom_word_dict, this_sentence )
                continue

            if tokenise_idiom :
                this_sentence = this_sentence.replace( idiom_phrase, idiom_tokeniser( idiom_phrase ) ) 


            original. append( this_sentence  ) 
            correct.  append( this_correct   ) 
            incorrect.append( this_incorrect )
            sent_idioms.append( idiom_phrase )


    assert len( original ) == len( correct ) == len( incorrect ) == len( sent_idioms )
    
    outdata = [ [ original[ i ], correct[ i ], incorrect[ i ], sent_idioms[ i ] ] for i in range( len( original ) ) ]
    outfile_name = os.path.join( out_location, 'trainToPredict.csv' ) 
    with open( outfile_name, 'w' ) as csvfile :
        writer = csv.writer( csvfile )
        writer.writerows( [ [ 'original', 'correct', 'incorrect', 'idiom' ] ] + outdata )
    print( "Wrote: ", outfile_name )
    
    outdata = [ [ original[ i ], sent_idioms[ i ], 1 ] for i in range( len( original ) ) ]
    outfile_name = os.path.join( out_location, 'trainSentsForIdiomClassification.csv' ) 
    with open( outfile_name, 'w' ) as csvfile :
        writer = csv.writer( csvfile )
        writer.writerows( [ [ 'sentence1', 'sentence2', 'label' ] ] + outdata )
    print( "Wrote: ", outfile_name )
    
    return



if __name__ == '__main__' :

    out_location   = 'trainData'

    ## Should always be False!!!
    ##   We do this later (in combineCreate) based on "select" and "all"
    tokenise_idiom = False


    if len( sys.argv ) < 2 :
        raise Exception( "Require language (EN, PT) as arg)" )
    
    language = sys.argv[1]
    assert language.lower() in [ 'en', 'pt' ]

    json_location = '../../../TaskIndependentData/' + language.lower() + '_TaskIndependentData.json'
    # [ 'train_one_shot', 'train_few_shot', 'train_zero_shot' ]

    print( "WARNING: Using Few Shot data - edit here to change to one_shot" )
    time.sleep( 1 ) 
    datasets = [ 'train_few_shot', 'train_zero_shot' ] 
    params = {
        'json_location'   : json_location                           ,
        'out_location'    : out_location                            ,
        'datasets'        : datasets                                ,
        'force_idiomatic' : False                                   ,
        'tokenise_idiom'  : tokenise_idiom                          ,
    }

    Path( params[ 'out_location' ] ).mkdir(parents=True, exist_ok=True)
    
    create_data( **params ) 
