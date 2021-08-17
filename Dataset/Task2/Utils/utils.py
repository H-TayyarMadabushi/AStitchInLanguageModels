import re
import csv

from collections   import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenise_idiom( phrase ) :
    
    return 'ID' + re.sub( r'[\s|-]', '', phrase ).lower() + 'ID'

if __name__ == '__main__' :
    print( tokenise_idiom( 'big fish' ) )
    print( tokenise_idiom( 'alta-costura' ) )
    print( tokenise_idiom( 'pastoralemã' ) )
    assert tokenise_idiom( 'big fish' ) == 'IDbigfishID'
    assert tokenise_idiom( 'alta-costura' ) == 'IDaltacosturaID'
    assert tokenise_idiom( 'pão-duro' ) == 'IDpãoduroID'
    assert tokenise_idiom( 'pastoralemão' ) == 'IDpastoralemãoID'
    print( "All good" )
    

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


def _load_csv( location, delimiter, is_gz=False ) :

    if delimiter is None :
        with open( location ) as infile :
            data = infile.read().lstrip().rstrip().split( '\n' )
            header = data[0]
            data   = data[1:]
        return header, data
            
    
    header = None
    data   = list()

    csvfile = reader = None
    if is_gz : 
        csvfile = gzip.open( location, 'rt', encoding='utf8' ) 
        reader = csv.reader( csvfile, delimiter=delimiter, quoting=csv.QUOTE_NONE )
    else : 
        csvfile = open( location ) 
        reader = csv.reader( csvfile, delimiter=delimiter )
    for row in reader :
        if header is None :
            header = row
            continue
        data.append( row )
    return header, data        
    
