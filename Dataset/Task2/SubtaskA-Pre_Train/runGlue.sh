## EN
model='edwardgowsmith/xlnet-base-cased-train-from-dev-and-test-short-best'

## PT 
model='harish/PT-UP-xlmR-OneShot-FalseTrue-0_2_BEST'

python ../Utils/run_glue.py \
       --model_name_or_path $model \
       --do_predict \
       --max_seq_length 128 \
       --output_dir models-no-git/trainData/PT/v3-dev-test-all/predict/ \
       --seed 42 \
       --test_file       models-no-git/trainData/PT/v3-dev-test-all/classification_sents.csv \
       --train_file      ../data/PT/v3/SentenceClassificationData/FalseTrue-0/train_from_dev_short.csv \
       --validation_file ../data/PT/v3/SentenceClassificationData/FalseTrue-0/train_from_dev_short.csv \
       --ignore_data_skip \
       --overwrite_output_dir 	
