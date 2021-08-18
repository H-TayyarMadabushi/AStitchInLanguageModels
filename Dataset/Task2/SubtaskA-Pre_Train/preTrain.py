import sys
import os
from datasets     import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

if len( sys.argv ) < 3 :
    print( "Require language and (select OR all)" )
    sys.exit()

language   = sys.argv[1]
train_what = sys.argv[2]

assert train_what in [ 'select', 'all' ]
assert language.upper() in [ 'EN', 'PT' ]

epoch                    = 5

model_location = save_to = None
train_eval_data_location = '../' + language.upper() + '_Pre-Train_Data/'

## These must be tokenized models
if language == 'EN' :
    model_location           = 'models-no-git/models/v3/BERT/bert-base-cased/'
    save_to                  = 'models-no-git/models/v3/BERT/pt-e5-' + train_what + '/'
elif language == 'PT' : 
    model_location           = 'models-no-git/models/v3/mBERT/bert-base-multilingual-cased/'
    save_to                  = 'models-no-git/models/v3/mBERT/pt-e' + str(epoch) + '-' + train_what + '/'


print( "train_eval_data_location:", train_eval_data_location )
print( "model_location:", model_location )
print( "save_to:", save_to )

sys.stdout.flush()

os.makedirs( save_to )

datasets       = load_dataset(
    "text",
    data_files = {
        "train"     : os.path.join( train_eval_data_location, train_what + "_train.txt" ),
        "validation": os.path.join( train_eval_data_location, train_what + "_eval.txt"  ),
    }
)

tokenizer      = AutoTokenizer.from_pretrained(
    model_location         , 
    use_fast       = False ,
    max_length     = 510   ,
    force_download = True  ,
)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True )
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=2, remove_columns=["text"] )

model         = AutoModelForMaskedLM.from_pretrained( model_location )

data_collator = DataCollatorForLanguageModeling( tokenizer=tokenizer, mlm_probability=0.15 )

training_args = TrainingArguments(
    save_to ,
    per_device_train_batch_size = 4,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=float( epoch ),
    save_strategy="epoch",
)

#     save_total_limit=3,

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

save_to                  = os.path.join( save_to, 'pretrained' )
os.makedirs( save_to )


trainer.save_model( save_to )
tokenizer.save_pretrained( save_to )
print( "Saved: ", save_to )
