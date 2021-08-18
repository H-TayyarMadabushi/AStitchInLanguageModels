import os
import re
import sys
import csv
import copy
import json
import random

from collections import defaultdict

sys.path.append( '../Utils/' )

from utils import tokenise_idiom as idiom_tokeniser
from utils import match_idioms, create_idiom_word_dict


random.seed( 42 ) 


def create_data( json_location, out_location, dataset, force_idiomatic, tokenise_idiom, idiom_list ) :

    assert not force_idiomatic 

    with open( json_location, encoding='utf-8') as fh:
        data = json.load(fh)
    data = data[ dataset ]

    # data = pickle.load( open( pickle_location, 'rb' ) )[ dataset ]
        
    correct_incorrect_pairs = list()
    sims        = list()
    new_sents   = list()
    ignored     = list()
    all_sents   = list()
    sent_idioms = list() # Idioms per sent
    idioms      = list() # All idioms
    for idiom in data :

        this_idiom = idiom[0][1]

        if not idiom_list is None :
            if not this_idiom.lower() in idiom_list :
                print( "WARNING: Ignoring {} based on idiom_list".format( idiom ) )
                continue
        
        idioms.append( this_idiom.lower().lstrip().rstrip() )
        
        meanings   = idiom[0][2:8]
        del_meanings = [ 'None', 'Meta Usage', 'Proper Noun' ] 
        ## del_meanings = [ 'None', 'Meta Usage' ]
        new_meanings = list()
        for meaning in meanings : 
            if meaning in del_meanings : 
                continue
            new_meanings.append( meaning ) 
        meanings = new_meanings

        for sent in idiom :
            all_sents.append( sent ) 
            label = sent[9] 
            if not label in meanings :
                ignored.append( sent ) 
                continue
            this_sentence   = sent[11]

            if not this_idiom in this_sentence :
                ignored.append( sent ) 
                continue

            this_correct    = re.sub( this_idiom, label, this_sentence, flags=re.I )
            # if this_correct == this_sentence :
            #     import pdb; pdb.set_trace()
            assert this_correct != this_sentence

            idiom_phrase    = sent[1]
            # idiom_words     = idiom_phrase.split()
            # idiom_word_dict = defaultdict( list )
            # idiom_word_dict[ idiom_words[0] ].append( idiom_words[1] )
            idiom_word_dict = create_idiom_word_dict( [ idiom_phrase ] )
            matched_idioms = match_idioms( idiom_word_dict, this_sentence )

            if len( matched_idioms ) == 0 :
                # print( "NO IDIOM!"  )
                # print( idiom_phrase )
                # print( this_sentence )
                ignored.append( sent ) 
                continue

            this_sentence_tokenised = re.sub( re.escape(idiom_phrase), idiom_tokeniser( idiom_phrase ), this_sentence, flags=re.I )
            assert this_sentence_tokenised != this_sentence
            sims.append( [ this_sentence, this_sentence_tokenised, this_correct, 1, this_idiom, this_correct, 'None' ] )
            
            not_label      = copy.copy( meanings )
            not_label.remove( label )

            if len( not_label ) == 0 :
                continue

            for single_not_label in not_label : 
                this_incorrect = re.sub( this_idiom, single_not_label, this_sentence, flags=re.I)
                assert this_incorrect != this_correct
                assert this_incorrect != this_sentence 
                sims.append( [ this_sentence, this_sentence_tokenised, this_incorrect, 'None', this_idiom, this_correct, this_incorrect ] )
            
                correct_incorrect_pairs.append( [ this_correct, this_incorrect ] )
            

    assert ( len( sims ) - len( correct_incorrect_pairs ) ) == ( len( all_sents ) - len( ignored ) )

    sim_header = [ 'sent_idiom', 'sent_idiom_tokenised', 'sent_other', 'sim', 'idiom', 'correct', 'incorrect' ]
    assert all( [ (len( i ) == len( sim_header )) for i in sims ] )

    print( "Total {} examples for {}".format( len( sims ), dataset ) )
    
    assert not tokenise_idiom
    if not tokenise_idiom :

        header                  = [ 'sentence1', 'sentence2', 'sim' ]
        sim_classification_data = [ header ] + [ [ i[sim_header.index( 'correct' ) ], i[sim_header.index( 'incorrect' )] , 1 ] for i in sims ]

        header                    = [ 'sentence1', 'sentence2', 'label' ]
        idiom_classification_data = [ header ] + [ [ i[sim_header.index( 'sent_idiom' )] , i[sim_header.index( 'idiom' )] , 1 ] for i in sims ]

        sims = [ sim_header ] + sims 

        assert len( sim_classification_data ) == len( sims ) == len( idiom_classification_data )
        
        ## For all three
        outfile_name = os.path.join( out_location, dataset + '_dataForSimilarity.csv' ) 
        with open( outfile_name, 'w' ) as csvfile :
            writer = csv.writer( csvfile )
            writer.writerows( sim_classification_data )
        print( "Wrote: ", outfile_name )

        outfile_name = os.path.join( out_location, dataset + '_rawSimData.csv' ) 
        with open( outfile_name, 'w' ) as csvfile :
            writer = csv.writer( csvfile )
            writer.writerows( sims )
        print( "Wrote: ", outfile_name )
        

        ## For Select Token
        outfile_name = os.path.join( out_location, dataset + '_dataForIdiomClassification.csv' ) 
        with open( outfile_name, 'w' ) as csvfile :
            writer = csv.writer( csvfile )
            writer.writerows( idiom_classification_data )
        print( "Wrote: ", outfile_name )
            
    return



if __name__ == '__main__' :


    print( """
WARNING: This script is not expected to be run. It is here for completeness only. 
Please note that the random seed might generate a new dataset for you making your results not comparable to those reported. 
To ensure that results are comparable, please start with the data in (EN/PT)/NoResults/

If you REALLY want to run this script, remove "sys.exit()" below.

""" )

    sys.exit()
        

    if len( sys.argv ) < 2 :
        raise Exception( "Require language (EN, PT) as arg)" )
    
    language = sys.argv[1]
    assert language.lower() in [ 'en', 'pt' ]

    json_location = '../../TaskIndependentData/' + language.lower() + '_raw_data.json'

    ## tokenise_idiom is not used and should always by false.
    for out_location, tokenise_idiom in [ ( 'evalData', False ) ] : 
        os.makedirs( out_location )
        for dataset in [ 'dev', 'test' ] : 
            params = {
                'json_location'   : json_location                        ,
                'out_location'    : out_location                         ,
                'dataset'         : dataset                              ,
                'force_idiomatic' : False                                ,
                'tokenise_idiom'  : tokenise_idiom                       ,
                'idiom_list'      : None                                 ,
            }


            create_data( **params ) 
