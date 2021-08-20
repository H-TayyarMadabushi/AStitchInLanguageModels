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
	   "../../CreateEvaluationData/${language}/WithResults/evalData/"
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
	   "../../CreateEvaluationData/${language}/WithResults/evalDataAllTokenised/"
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
	   "../../CreateEvaluationData/${language}/WithResults/evalDataSelectTokenised/"
done

