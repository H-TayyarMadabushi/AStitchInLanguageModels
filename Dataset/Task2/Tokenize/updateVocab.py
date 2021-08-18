import re
import os
import sys



# insert_vocab_file = '../EN_Pre-Train_Data/vocab_update.txt' 
# vocab_file_name   = 'models-no-git/models/v3/BERT/bert-base-cased/vocab.txt'

insert_vocab_file = '../PT_Pre-Train_Data/vocab_update.txt'
vocab_file_name   = 'models-no-git/models/v3/mBERT/bert-base-multilingual-cased/vocab.txt'



backup            = vocab_file_name.split( '.txt' )[0] + '.bk'

os.system( 'cp ' + vocab_file_name + ' ' + backup )

insert_vocab = open( insert_vocab_file ).read().lstrip().rstrip().split( '\n' )

new_vocab = list()
with open( vocab_file_name, 'r' ) as vocab_file: 
  for token in vocab_file : 
    token = token.lstrip().rstrip()
    if len( insert_vocab ) > 0 and not re.match( r'\[unused\d+\]', token ) is None :
      token = insert_vocab.pop( 0 )
    new_vocab.append( token )

assert len( insert_vocab ) == 0
with open( vocab_file_name, 'w' ) as vocab_file: 
  vocab_file.write( '\n'.join( new_vocab ) )

print( "Updated: ", vocab_file_name )
