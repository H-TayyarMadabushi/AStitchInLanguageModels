import re
import os
import sys
import csv
import pickle
import random
random.seed( 42 )

from tqdm          import tqdm
from collections   import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

sys.path.append( '.' )

from tokeniseIdiom import tokenise_idiom

def match_idioms( idiom_word_dict, sentence ) :

    sentence_words = word_tokenize( sentence )
    new_sentence_words = list()
    for word in sentence_words :
        if not re.match( r'^\'\w', word ) is None :
            new_sentence_words.append( "'" )
            word = re.sub( r'^\'', '', word )
            new_sentence_words.append( word )
            continue
        new_sentence_words.append( word )
    sentence_words = new_sentence_words

    matched_idioms = list()
    for index in range( len( sentence_words ) - 1 ) :
        this_word   = sentence_words[ index ].lower() 
        idiom_words = idiom_word_dict[ this_word ]
        if len( idiom_words ) == 0 :
            continue
        next_word = sentence_words[ index + 1 ].lower()
        for idiom_word_2 in idiom_words :
            if idiom_word_2.lower() == next_word or idiom_word_2 + 's' == next_word or idiom_word_2 == '*' :
                matched_idioms.append( this_word + ' ' + idiom_word_2  )
                
    return matched_idioms

def create_idiom_word_dict( idioms ) : 
    
    idiom_word_dict = defaultdict( list )
    for idiom in idioms :
        split_idiom = idiom.split()
        if len( split_idiom ) == 2 : 
            word1, word2 = split_idiom
        elif len( split_idiom ) == 1 : 
            word1 = split_idiom[0]
            word2 = '*'
        else : 
            raise Exception( "Cannot handle length!" )

        idiom_word_dict[ word1 ].append( word2 ) 
        
    return idiom_word_dict

def _load_single_dataset_sents( location ) :

    sents = list()
    with open( location ) as csvfile :
        reader = csv.reader( csvfile )
        for row in reader :
            sents.append( row[0] )
    return sents

def _load_dataset_sents( dataset_sents_info ) :

    files = [ 'dev_sents.csv', 'test_sents.csv', 'train_few_shot_sents.csv', 'train_zero_shot_sents.csv' ]

    sents = list()
    for file_name in files :
        file_location = os.path.join( dataset_sents_info, file_name )
        sents += _load_single_dataset_sents( file_location )
        
    return sents 

def sent_in_dataset( dataset_sents, sent ) :

    sent = ''.join( sent.lower().split() )
    for dataset_sent in dataset_sents :
        dataset_sent = ''.join( dataset_sent.lower().split() )
        if  dataset_sent in sent or sent in dataset_sent :
            return True
    return False

def filter( train_dev_location, data_location, out_location, datasets, dataset_sents_info, idioms=None, limit_count=None ) :

    dataset_sents = _load_dataset_sents( dataset_sents_info )
    

    if idioms is None : 
        idiom_sents = dict()

        for dataset in datasets :
            header      = None
            with open( os.path.join( train_dev_location, dataset ) ) as csvfile : 
                reader = csv.reader(csvfile) 
                for row in reader:
                    if header is None :
                        header = row
                        continue
                    label = int( row[0] )
                    sent  = row[1]
                    idiom = row[2]
                    if not idiom in idiom_sents.keys() :
                        idiom_sents[ idiom ] = { 0 : list(), 1 : list() }
                    idiom_sents[ idiom ][ label ].append( sent )
        idioms = [i.lower() for i in list( idiom_sents.keys() ) ]

    # counts = pickle.load( open( 'data/processCCNNews-status.pk3', 'rb' ) )[ 'counts' ]
    # for idiom in idioms :
    #     print( idiom, "-->", counts [ idiom ] )
    # sys.exit()

    #Write vocab
    idioms_write = sorted( [ [ tokenise_idiom( i ) ] for i in idioms ] )
    outfile_name = os.path.join( out_location, 'vocab_update.txt' )
    with open( outfile_name, 'w' ) as csvfile :
        writer = csv.writer(csvfile)
        writer.writerows( idioms_write )
        print( "Wrote: ", outfile_name )

    idiom_word_dict = create_idiom_word_dict( idioms ) 

    data_files = [f for f in os.listdir( data_location ) if os.path.isfile(os.path.join(data_location, f))]

    line_number           = 0
    documents_no_replace  = list()
    documents_all_replace = list()
    classification_sents  = list()
    included_counts       = defaultdict( int ) 
    for data_file in tqdm( data_files ) :
        data_file = os.path.join( data_location, data_file )
        data      = open( data_file, 'r', encoding='utf-8', errors='ignore' ).read()
        for doc in data.split( '\n--DocBreak--\n' ) :

            this_doc = list()
            for line in doc.split( '\n' ) :
                line = line.lstrip().rstrip()
                if len( line ) < 5 :
                    continue
                if line[0] == '*' :
                    continue;
                this_doc += [ i for i in sent_tokenize( line ) if len( i ) > 5 and len( i.split() ) > 3 ]
            this_doc = [ i.replace( '**', '' ) for i in this_doc ]
            this_doc = [ i.replace( '_', '' ) for i in this_doc ]
            this_doc = [ i.lstrip().rstrip() for i in this_doc ]
            this_doc = [ re.sub( r'\s+', ' ', i ) for i in this_doc ]

            all_doc         = list()
            replaced_doc    = list()
            replaced        = False
            this_line       = 0
            for sent in this_doc :
                original_sent = sent
                if len( sent.split() ) > 500 :
                    continue
                if any( [ ( i in sent ) for i in idioms ] ) :
                    if sent_in_dataset( dataset_sents, sent ) :
                        print( "Found sent in dataset: ", sent, flush=True )
                        continue

                    matched_idioms = match_idioms( idiom_word_dict, sent )
                    matched_idioms = [ i.lower() for i in list( set( matched_idioms ) ) ]
                    no_new_idiom = True
                    if not limit_count is None : 
                        for matched_idiom in matched_idioms :
                            if not limit_count is None and included_counts[ matched_idiom ] != limit_count :
                                no_new_idiom = False
                                included_counts[ matched_idiom ] += 1
                    else :
                        no_new_idiom = False        
                    if no_new_idiom :
                        break
                    
                    if len( matched_idioms ) > 0 :
                        replaced = True
                        for matched_idiom in matched_idioms :
                            classification_sents.append( [ line_number + this_line, sent, matched_idiom, 1 ] )
                        for matched_idiom in matched_idioms :
                            sent = sent.replace( matched_idiom, tokenise_idiom( matched_idiom ) )
                    replaced_doc.append( sent ) 
                else :
                    replaced_doc.append( sent )
                    
                all_doc.append( original_sent )
                    
                this_line += 1
            if replaced :
                documents_no_replace.append( all_doc )
                documents_all_replace.append( replaced_doc )
                line_number += this_line + 1
            assert len( replaced_doc ) == len( all_doc )

            got_all_data = True
            for idiom in idioms :
                if not limit_count is None and included_counts[ idiom.lower() ] < limit_count :
                    got_all_data = False
                    break
            if not limit_count is None and got_all_data :
                break


    for outfile_name, data in [
            [ os.path.join( out_location, 'no_replace_data.txt' ) , documents_no_replace  ],
            [ os.path.join( out_location, 'all_replace_data.txt' ), documents_all_replace ]
    ] : 
        with open( outfile_name, 'w' ) as outfile :
            # Header
            outfile.write( 'text' + '\n' )
            for doc in data :
                for sent in doc :
                    outfile.write( sent + "\n" )
                outfile.write( "\n" )
        print( "Wrote: ", outfile_name )

    outfile_name = os.path.join( out_location, 'classification_sents.csv' )
    with open( outfile_name, 'w' ) as csvfile :
        writer = csv.writer(csvfile)
        writer.writerows( [ [ 'sent_id', 'sentence1', 'sentence2', 'label' ] ] + classification_sents )
        print( "Wrote: ", outfile_name )

    return


def _get_dataset_idioms( datasets, data_path ) :

    idiom_data = pickle.load( open( os.path.join( data_path, 'idioms.pk3' ), 'rb' ) )
    idioms     = list()
    for dataset in datasets :
        dataset = dataset.split( '.csv' )[0].lower()
        this_dataset_idioms = idiom_data[ dataset ]
        idioms += this_dataset_idioms

    idioms = list( set( idioms ) ) 
    print( "Picked {} idioms.".format( len( idioms ) ) )
    assert len( idioms ) > 0
    return idioms
                                           
    
if __name__ == '__main__' :


    if len( sys.argv ) < 2 :
        print( "Require languages (EN|PT)" )
        sys.exit()
    language = sys.argv[1]
    
    train_dev_location = '../../Task1/SubTaskA/' + language + '/ContextIncluded_IdiomIncluded/'
    dataset_sents_info = '../../Task1/SubTaskA/' + language + '/'
    
    params = {
        'train_dev_location' : train_dev_location ,
        'data_location'      : 'output-no-git/'   ,
        'out_location'       : 'output-no-git/' + language + '/' ,
        'datasets'           : [ 'dev.csv', 'test.csv' ], 
        'dataset_sents_info' : dataset_sents_info  ,
        'idioms'             : 'dataset', 
        'limit_count'        : None , 
    }

    from pprint import pprint
    print( "PARAMS: " )
    pprint( params )
    

    ## Load up idiom list if required
    if type( params[ 'idioms' ] ) == str and params[ 'idioms' ].lower() == 'dataset' :
        # data_path = '../sentenceTransformers/evalData/'
        data_path = params[ 'train_dev_location' ]
        params[ 'idioms' ] = _get_dataset_idioms( params[ 'datasets' ], data_path )

        from pprint import pprint
        print( "UPDATED PARAMS: " )
        pprint( params )
        
    
    os.makedirs( params[ 'out_location' ] )
    filter( **params ) 
