# Predict lncRNA-drug associations based on graph neural networks 

## 1. Introduction
This reository contains source code and data for the paper "*Predict lncRNA-drug associations based on graph neural networks*".
## 2. Installation
This python code depends on the following packages. You must have them installed before running this code.

conda install pytorch==1.8.1 torchvision==0.16.0 -c pytorch  
pip install pandas 
pip install numpy  
pip install scipy  
pip install torch_geometric  
pip install networkx  
pip install pickle  
pip install scikit-learn

## 3. Usage
### 3.1 Data
The all datasets are freely downloaded from <https://pan.baidu.com/s/1fF57IzpflJNxbJKLJLlCWg?pwd=l7xa>
### 3.2 Training model
Run Main.py, and set train parameter to TRUE and use-feature parameter to TRUE, specifying the dataset to train the model.
### 3.3 Predicting LDAs
Run Main.py and set the parameter train to False and the parameter use-features to TRUE, using the model.pth file generated in 3.2 to specify the data set to predict potential LDAs.
