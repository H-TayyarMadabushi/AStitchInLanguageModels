# Extended Noun Compound Senses Dataset

This dataset provides all possible senses associated with noun compounds in both [Portuguese](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Extended_Noun_Compound_Senses_Dataset/PT_Extended_Noun_Compound_Senses_Dataset.csv) and [English](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Extended_Noun_Compound_Senses_Dataset/EN_Extended_Noun_Compound_Senses_Dataset.csv). 

This dataset is a subset of the AStitchInLanguageModels dataset. 

The dataset contains the following columns: 
* Multiword Expression	(E.g Elbow room)
* Literal Meaning	(Literal meaning if there is one, else a literal interpretation)
* Non-Literal Meaning 1	(Non-literal meaning if it exists "None" otherwise")
* Non-Literal Meaning 2	(Non-literal meaning if it exists "None" otherwise")
* Non-Literal Meaning 3	(Non-literal meaning if it exists "None" otherwise")
* Data Split (Train/Dev/Test)


This data differs from previous MWE sense datasets in that: 
 * it provides all possible senses,
 * we ensure that meanings provided are as close to the original phrase as possible to ensure that this dataset is an adversarial dataset, 
 * we highlight purely compositional noun compounds. 
