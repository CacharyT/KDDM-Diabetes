#KDDM-Diabetes Dataset
#Contributors: Cachary Tolentino and Ian Valiante
#10/5/24

#imports
#import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy.stats
from sklearn import *
#import seaborn as sns


#Import dataset
df = pd.read_csv("diabetes_dataset00.csv")
df.head()
df.info()


#Data Cleaning (Missing data and Duplicate Data)

# Num of missing values
nanCount = df.isnull().sum()
print("Number of NaN values:")
print(nanCount)

#Num of duplicated values
dupCount = df.duplicated().sum()
print("\nNumber of duplicated values: ",dupCount)




#Data Preprocessing
#Options: Aggregation, Sampling, Dimensionality Reduction, Feature subset selection, Feature creation, Discretization, Binarization, and Variable Tranformation
#Do: Sampling and Feature Subset Selection
features = df.columns
print(features)
print(len(features))
print("Population Size: ", len(df))

#Sampling
sample_with_replacement = df.sample(frac = .01, replace = True)
sample_without_replacement = df.sample(frac = .01)


#Visualization
#plt.bar(sample_with_replacement["Age"])


#Data Mining


#Data Postprocessing