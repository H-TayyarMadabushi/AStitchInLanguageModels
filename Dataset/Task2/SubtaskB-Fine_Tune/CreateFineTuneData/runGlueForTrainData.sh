
set -e
## Use a model trained on the one-shot data from Task 1 here. 

## En
model_name_or_path='edwardgowsmith/xlnet-base-cased-train-from-dev-and-test-short-best'

## PT
## model_name_or_path='harish/PT-UP-xlmR-OneShot-FalseTrue-0_2_BEST'


python ../../Utils/run_glue.py \
       --model_name_or_path $model_name_or_path \
       --do_predict \
       --max_seq_length 128 \
       --output_dir      trainData/predictions \
       --seed 42 \
       --test_file       trainData/trainSentsForIdiomClassification.csv   \
       --train_file      ../../../Task1/SubTaskA/EN/ContextExcluded_IdiomIncluded/train_one_shot.csv \
       --validation_file ../../../Task1/SubTaskA/EN/ContextExcluded_IdiomIncluded/dev.csv \
       --ignore_data_skip \
       --overwrite_output_dir 	

