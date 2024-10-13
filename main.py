#KDDM-Diabetes Dataset
#Contributors: Cachary Tolentino and Ian Valiante
#10/5/24

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from scipy import stats
from sklearn import *
import seaborn as sns 
import random


#Import dataset----------------------------------------------------------------------------------------------------------------
df = pd.read_csv("diabetes_dataset00.csv")
df.head()
df.info()


#Data Cleaning (Missing data, Duplicate Data, Noisy Data, and Outliers)--------------------------------------------------------------------------------


# Num of missing values
nanCount = df.isnull().sum()
print("Number of NaN values:")
print(nanCount)


#Num of duplicated values
dupCount = df.duplicated().sum()
print("\nNumber of duplicated values: ", dupCount)


#Boxplotting to find any noise data and outliers (Numeric Features only)
numeric_features = df.select_dtypes(include = np.number).columns.tolist()

for column in numeric_features:
    plt.boxplot(df[column].tolist())
    plt.title(column)
    plt.xlabel("Data")
    plt.ylabel("Values")
    plt.show()

#Removing outliers (using IQR)
q1_waist, q3_waist = np.percentile(df['Waist Circumference'], [25,75])
iqr_waist = q3_waist - q1_waist
lower_bound_waist = q1_waist - 1.5 * iqr_waist
upper_bound_waist = q3_waist + 1.5 * iqr_waist

q1_pulmonary, q3_pulmonary = np.percentile(df['Pulmonary Function'], [25,75])
iqr_pulmonary = q3_pulmonary - q1_pulmonary
lower_bound_pulmonary = q1_pulmonary - 1.0 * iqr_pulmonary #decreased threshold
upper_bound_pulmonary = q3_pulmonary + 1.0 * iqr_pulmonary #decreased threshold

df = df[(df['Waist Circumference'] > lower_bound_waist) & (df['Waist Circumference'] < upper_bound_waist)]
df = df[(df['Pulmonary Function'] > lower_bound_pulmonary) & (df['Pulmonary Function'] < upper_bound_pulmonary)]


plt.boxplot(df['Waist Circumference'])
plt.title('Waist Circumference - Removed Outliers')
plt.xlabel("Data")
plt.ylabel("Values")
plt.show()

plt.boxplot(df['Pulmonary Function'])
plt.title('Pulmonary Function - Removed Outliers')
plt.xlabel("Data")
plt.ylabel("Values")
plt.show()



#Data Preprocessing (Sampling, Dimensionality Reduction, Feature Subset Selection, and Discretization)----------------------------------------------------------------------


#Sampling (with and without replacement; sample size = 10% of population)
print("Original population size: ", len(df))
sampled_df = df.sample(frac = 0.10, replace = True, random_state = 50) #keep random_state to ensure same values each time
print("New population size: ", len(sampled_df))


#Bootstrapping to determine represenativity of original sample (only numerics)
sampling_df = df.select_dtypes(include = np.number).columns.tolist()

sampled_means = []
for column in sampling_df:  #Finds means for the original sample
    sampled_means.append(sampled_df[column].mean())


def bootstrap_mean(df, columnName): #Bootstrapping function for means
    bootstrapped = df.sample(frac = 0.10, replace = True)
    return  bootstrapped[columnName].mean()

bootstrapped_means = []

for column in sampling_df:  #Finds means for each bootstrapped sample
    current_means = [bootstrap_mean(sampled_df, column) for i in range(10001)]
    bootstrapped_means.append(sum(current_means)/len(current_means))


for i in range(len(bootstrapped_means)):
    print(sampling_df[i]," Original - Bootstrapped:", sampled_means[i], " - ", bootstrapped_means[i])



#Feature Subsetting/Selection
key_features = [
    'Target', 'Genetic Markers', 'Autoantibodies', 'Family History',
    'Environmental Factors', 'Insulin Levels', 'Age', 'BMI','Physical Activity'
]

bonus_features = [
       'Dietary Habits', 'Blood Pressure',
       'Cholesterol Levels', 'Waist Circumference', 'Blood Glucose Levels',
       'Ethnicity', 'Socioeconomic Factors', 'Smoking Status',
       'Alcohol Consumption', 'Glucose Tolerance Test', 'History of PCOS',
       'Previous Gestational Diabetes', 'Pregnancy History',
       'Weight Gain During Pregnancy', 'Pancreatic Health',
       'Pulmonary Function', 'Cystic Fibrosis Diagnosis',
       'Steroid Use History', 'Genetic Testing', 'Neurological Assessments',
       'Liver Function Tests', 'Digestive Enzyme Levels', 'Urine Test',
       'Birth Weight', 'Early Onset Symptoms'
]

numeric_features = df.select_dtypes(include = np.number).columns.tolist()


#Discretization
sampled_df['Age'] = pd.cut(sampled_df['Age'], bins = [0, 12, 20, 60, sampled_df['Age'].max()], labels = ['Child', 'Teen', 'Adult', "Elderly"])
print(sampled_df['Age'])

sampled_df['BMI'] = pd.cut(sampled_df['BMI'], bins = [0, 18.5, 25, 30, sampled_df['BMI'].max()], right = False, labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Obese'])
print(sampled_df['BMI'])


#Data Mining-----------------------------------------------------------------------------------------------------------------------


#Data Postprocessing---------------------------------------------------------------------------------------------------------------