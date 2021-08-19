set -e
## Use a model trained on the one-shot data from Task 1 here. 

## En
model_name_or_path='edwardgowsmith/xlnet-base-cased-train-from-dev-and-test-short-best'

## PT
## model_name_or_path='harish/PT-UP-xlmR-OneShot-FalseTrue-0_2_BEST'


folder='evalData'

for dataset in dev test
do
    test_file="${folder}/${dataset}_dataForIdiomClassification.csv"
    outlocation="${folder}/predict_${dataset}"

    python ../Utils/run_glue.py \
	   --model_name_or_path $model_name_or_path \
	   --do_predict \
	   --max_seq_length 128 \
	   --output_dir      $outlocation \
	   --seed 42 \
	   --test_file       $test_file \
	   --train_file  ../../Task1/SubTaskA/EN/ContextExcluded_IdiomIncluded/train_one_shot.csv \
	   --validation_file ../../Task1/SubTaskA/EN/ContextExcluded_IdiomIncluded/dev.csv \
	   --ignore_data_skip \
	   --overwrite_output_dir 	

done



