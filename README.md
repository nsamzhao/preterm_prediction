# DREAM Preterm Challenge

## Dependencies

You will need some standard numerica/scientific packages in Python, such as `pandas`, `numpy`, `scipy`, `scikit-learn`, and `matplotlib`. If you have Anaconda, it is likely these are already installed.

You will likely need to install these manually: 
1. `scanpy` - Python package to preprocess single-cell data, similar to Seurat in R.
2. `PyTorch` - Deep Learning framework.
3. `plotnine` - Python package equivalent to `ggplot2` in R.

## Data Folder Structure
**Note/Update**: Git restricts each data file to be less than 200 MB, and generally not recommended to upload data files. Since our data files are small, I am uploading them so you may skip some of the preprocessing.  
Now data are at [google drive](https://drive.google.com/open?id=1uLt_aotV0Jq42iAATh2nPEsJuLErGb2h)!
```
|-- data
|   |-- raw
|       |-- preterm_annotation.csv # This is from the annotation object from eset_SC2_v20.RData
|       |-- preterm_data.csv # This is from the data object from eset_SC2_v20.RData
|   |-- processed # Folder containing training data for Classifier #1 and Classifier #2, as well as X_test.csv (all genes)
|   |-- feature_selected # Folder containing training data and testing data after selecting for 100 genes
|   |-- torch # Folder containing PyTorch objects for training and testing
```

## How to Run

### Linear Models
Simply run:
```
python logistic_regression.py
```

Outputs will be shown in the terminal. 

### Deep Learning Models
To train:
```
python train.py -c configs/sPTD_config.json # or configs/PPROM_config.json
```
Feel free to play around with the configuration files to test different hyperparameters.

To run inference (and generate prediction file):
```
python test.py -r PATH TO SAVED MODEL
```
**Note**: I have not written the inference code yet. 

Outputs will be saved to `saved` folder (which will automatically be created if you don't already have one). 

## Visualizing Results

Visualization in Tensorboard:
```
cd saved/log/PATH TO RESULT/
tensorboard --logdir .
```

## Development

Main development steps are all included in [notebook](https://github.com/hojaeklee/preterm/blob/master/notebook/select_features_develop_models.ipynb)  
Including:  
- feature selection steps
- develpment of Random Forest, SVM, and Logistic Regression
- the developed ensemble classifier from those three trained classifiers
