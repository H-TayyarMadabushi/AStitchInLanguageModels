import os
import sys

from transformers import AutoTokenizer

model_location           = sys.argv[1] 

tokenizer      = AutoTokenizer.from_pretrained(
    model_location         , 
    use_fast       = False ,
    max_length     = 510   ,
    force_download = True
)


some_pass = False
if tokenizer.tokenize('This is a IDancienthistoryID')[-1] == 'IDancienthistoryID' :
    print( tokenizer.tokenize('This is a IDancienthistoryID'), flush=True )
    some_pass = True
    
if tokenizer.tokenize( 'This is a IDcolégiomilitarID'     )[-1] == 'IDcolégiomilitarID'    :
    print( tokenizer.tokenize( 'This is a IDcolégiomilitarID' ) )
    some_pass = True
assert some_pass
print( "All good", flush=True )

