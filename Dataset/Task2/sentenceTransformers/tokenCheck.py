import os
import sys

from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util


model_location           = sys.argv[1] 

model = SentenceTransformer( model_location )


some_pass = False
if model.tokenizer.tokenize('This is a IDancienthistoryID')[-1] == 'IDancienthistoryID' :
    print( model.tokenizer.tokenize('This is a IDancienthistoryID'), flush=True )
    some_pass = True
    
if model.tokenizer.tokenize( 'This is a IDcolégiomilitarID'     )[-1] == 'IDcolégiomilitarID'    :
    print( model.tokenizer.tokenize( 'This is a IDcolégiomilitarID' ) )
    some_pass = True
assert some_pass
print( "All good", flush=True )
