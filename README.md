# Dog Breed Identifacation

    A example of pytorch custom dataset.

    Kaggle url: https://www.kaggle.com/competitions/dog-breed-identification


## Processed
1. Read image ids and breeds from labels.csv.
2. Create breed dictionary based on breed information.
3. Append "image pathname" or "None" to labels.csv based on the results of if image exist or not.
4. Append breed ids to labels.csv, which are the mapping result of breed information and dictionary.
5. Split image pathnames and breed ids to two parts respectivelly, first part for training, second part for validation.
6. Save these four group data as respective npy file.


## Data
    
    https://www.kaggle.com/c/dog-breed-identification/data

## Pretrained pytorch models

    https://www.kaggle.com/pvlima/pretrained-pytorch-models
