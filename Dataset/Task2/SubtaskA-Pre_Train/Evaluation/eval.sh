set -e

## Change if you've used your own model to create evaluation data. 
echo "Change Language to PT if running PT!"

language='EN' 
basefolder="../../CreateEvaluationData/${language}/WithResults/"

model='BERT'
if [ "$language" == "PT" ]
then
    model='mBERT'
fi


## Not making use of special tokens.
echo "bert-base-cased-tokenised, evalData"
python ../../Evaluation/task2Evaluation.py   \
       "output-no-git/$model/STModels/bert-base-cased-tokenised/v3-s2/" \
       "${basefolder}/evalData/"

echo "****************************************************************************************************"
## Make use of special tokens.
echo "bert-base-cased-tokenised, evalDataAllTokenised"
python ../../Evaluation/task2Evaluation.py   \
       "output-no-git/$model/STModels/bert-base-cased-tokenised/v3-s2/" \
       "${basefolder}/evalDataAllTokenised/"
    
echo "****************************************************************************************************"
echo "pt-e5-all evalDataAllTokenised"
python ../../Evaluation/task2Evaluation.py    \
       "output-no-git/$model/STModels/pt-e5-all/v3-s5/"  \
       "${basefolder}/evalDataAllTokenised/"

echo "****************************************************************************************************"
echo "pt-e5-select evalDataSelectTokenised"
python ../../Evaluation/task2Evaluation.py    \
       "output-no-git/$model/STModels/pt-e5-select/v3-s5/" \
       "${basefolder}/evalDataSelectTokenised/"

done
