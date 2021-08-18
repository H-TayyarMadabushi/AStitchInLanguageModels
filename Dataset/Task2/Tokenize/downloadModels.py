

# model_checkpoint = 'bert-base-cased' 
model_checkpoint = 'bert-base-multilingual-cased' 

# outdir = 'models-no-git/models/BERT/'

outdir = 'models-no-git/models/mBERT/'
outdir = outdir + model_checkpoint + '/'

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.save_pretrained( outdir )


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, truncation=True)
tokenizer.save_pretrained( outdir )


print( "Wrote to: ", outdir )
