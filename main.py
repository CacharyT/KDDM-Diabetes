#KDDM-Diabetes Dataset
#Contributors: Cachary Tolentino and Ian Valiante
#10/5/24

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy.stats
from sklearn import *
import seaborn as sns 


#Import dataset----------------------------------------------------------------------------------------------------------------
df = pd.read_csv("diabetes_dataset00.csv")
df.head()
df.info()


#Data Cleaning (Missing data and Duplicate Data)--------------------------------------------------------------------------------


# Num of missing values
nanCount = df.isnull().sum()
print("Number of NaN values:")
print(nanCount)


#Num of duplicated values
dupCount = df.duplicated().sum()
print("\nNumber of duplicated values: ", dupCount)


#Data Preprocessing (Sampling, Dimensionality Reduction, Feature Subsetting/Selections, and Outlier Detection)-------------------


#Dimensionality Reduction (Dropping Neurological Assessment and Genetic testing)
print("Pre-removal Features: \n", df.columns)
df.drop("Neurological Assessments", axis=1,inplace=True)
df.drop("Genetic Testing", axis=1,inplace=True)
print("Post-removal Features: \n", df.columns)



#sampling (with and without replacement; sample size = 10% of population)
print("Original population size: ", len(df))
with_df = df.sample(frac = 0.1, replace = True, random_state = 50) #keep random_state to ensure same values eachh time
print("New population size: ", len(with_df))


#Feature Subsetting/Selection
base_features = ['Insulin Levels','Family History','Early Onset Symptoms','Genetic Markers','Steroid Use History'] 

age_subset = with_df[[feature for feature in base_features + ['Age']]]
diet_health_subset = with_df[[feature for feature in base_features + ['Physical Activity','Dietary Habits']]]
target_subset = with_df[[feature for feature in base_features + ['Target']]]
auto_antibodies_subset = with_df[[feature for feature in base_features + ['Autoantibodies']]]


print("Age Feature subset:\n", age_subset)
print("Diet Habits/Physical Activity Feature subset:\n", diet_health_subset)
print("Target Feature subset:\n", target_subset)
print("Autoantibodies Feature subset:\n", auto_antibodies_subset)


#Outlier Detection (Visualization for each numeric feature involving numeric values then compute z-score to remove any outliers )
plt.boxplot(with_df['Insulin Levels'])
plt.title("Insulin Levels")
plt.show()

plt.boxplot(with_df['Age'])
plt.title("Age")
plt.show()


#Data Mining-----------------------------------------------------------------------------------------------------------------------


#Data Postprocessing---------------------------------------------------------------------------------------------------------------