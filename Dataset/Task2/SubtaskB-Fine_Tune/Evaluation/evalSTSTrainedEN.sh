set -e

language = 'EN'

echo "********************************************************************************"
echo "                         LANGUAGE: ${language}                                  "
echo "********************************************************************************"

### None
echo "********************************************************************************"
echo " NONE "
echo "********************************************************************************"

baselocation="output-no-git/SubtaskB/trainDataNotTokenised/0"
for epoch in $( seq 1 $epochs )
do

    python task2Evaluation.py \
	   "${baselocation}/${epoch}/" \
	   "../../CreateEvaluationData/EN/WithResults/${language}/evalData/"
done


### All
echo "********************************************************************************"
echo " All "
echo "********************************************************************************"
baselocation="output-no-git/SubtaskB/trainDataAllTokenised/0"
for epoch in $( seq 1 $epochs )
do

    python task2Evaluation.py \
	   "${baselocation}/${epoch}/" \
	   "../../CreateEvaluationData/EN/WithResults/${language}/evalDataAllTokenised/"
done

### Select
echo "********************************************************************************"
echo " Select "
echo "********************************************************************************"
baselocation="output-no-git/SubtaskB/trainDataSelectTokenised/0"
for epoch in $( seq 1 $epochs )
do

    python task2Evaluation.py \
	   "${baselocation}/${epoch}/" \
	   "../../CreateEvaluationData/EN/WithResults/${language}/evalDataSelectTokenised/"
done

