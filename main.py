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
import math


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

# for column in numeric_features:
#     plt.boxplot(df[column].tolist())
#     plt.title(column)
#     plt.xlabel("Data")
#     plt.ylabel("Values")
#     plt.show()

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

def plotOutliers(dfName):
    plt.boxplot(df[dfName])
    plt.title(f'{dfName} - Removed Outliers')
    plt.xlabel("Data")
    plt.ylabel("Values")
    plt.show()
    
# plotOutliers('Waist Circumference')
# plotOutliers('Pulmonary Function')



#Data Preprocessing (Sampling, Dimensionality Reduction, Feature Subset Selection, and Discretization)----------------------------------------------------------------------


#Sampling (with and without replacement; sample size = 10% of population)
print("Original population size: ", len(df))
sampled_df = df.sample(frac = 0.10, replace = True, random_state = 50) #keep random_state to ensure same values each time
print("New population size: ", len(sampled_df))


#Bootstrapping to determine represenativity of original sample (only numerics)
sampling_df = df.select_dtypes(include = np.number).columns.tolist()

# sampled_means = []
# for column in sampling_df:  #Finds means for the original sample
#     sampled_means.append(sampled_df[column].mean())


# def bootstrap_mean(df, columnName): #Bootstrapping function for means
#     bootstrapped = df.sample(frac = 0.10, replace = True)
#     return  bootstrapped[columnName].mean()

# bootstrapped_means = []

# for column in sampling_df:  #Finds means for each bootstrapped sample
#     current_means = [bootstrap_mean(sampled_df, column) for i in range(10001)]
#     bootstrapped_means.append(sum(current_means)/len(current_means))


# for i in range(len(bootstrapped_means)):
#     print(sampling_df[i]," Original - Bootstrapped:", sampled_means[i], " - ", bootstrapped_means[i])



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
# sampled_df['Age'] = pd.cut(sampled_df['Age'], bins = [0, 12, 20, 60, sampled_df['Age'].max()], labels = ['Child', 'Teen', 'Adult', "Elderly"])
# print(sampled_df['Age'])

# sampled_df['BMI'] = pd.cut(sampled_df['BMI'], bins = [0, 18.5, 25, 30, sampled_df['BMI'].max()], right = False, labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Obese'])
# print(sampled_df['BMI'])


#Data Analayis & Implementation-----------------------------------------------------------------------------------------------------------------------

#Decision Tree

#Criterion for decision (Gini Impurity)
def gini(labels):
    count = np.bincount(labels)
    probability = count / len(labels)
    return 1 - sum(prob ** 2 for prob in probability)

#Data splitting based on feature and threshold
def data_split(x, y, feature_index, value, categorical=False):
    left_x, left_y, right_x, right_y = [], [], [], []
    for val, row in enumerate(x):
        if categorical:
            if row[feature_index] == value:
                left_x.append(row)
                left_y.append(y[val])
            else:
                right_x.append(row)
                right_y.append(y[val])
        else:
            if row[feature_index] < value:
                left_x.append(row)
                left_y.append(y[val])
            else:
                right_x.append(row)
                right_y.append(y[val])
    return left_x, left_y, right_x, right_y

#Finding best split based on Gini
def gini_split(x,y):
    best_gini = float("inf")
    best_params = None
    for index in range(len(x[0])):

        unique_val = set(row[index] for row in x)
        categorical = all(isinstance(val, int) for val in unique_val) and len(unique_val) < 10

        for value in unique_val:
            left_x, left_y, right_x, right_y = data_split(x, y, index, value, categorical)
            if not left_y or not right_y:
                continue
            gini_val = (len(left_y) / len(y) * gini(left_y) + len(right_y) / len(y) * gini(right_y))
            if gini_val < best_gini:
                best_gini = gini_val
                best_params = (index, value, left_x, left_y, right_x, right_y)
    return best_params

#Creating deciison tree
def tree_builder(x, y, depth=0, max_depth = 10):
    if len(set(y)) == 1 or depth == max_depth:
        return {"label": max(set(y), key = y.count)} #Leaf node
    
    split = gini_split(x,y)
    if split is None:
        return {"label": max(set(y), key = y.count)} #Leaf node since no possible split

    feature_index, threshold, left_x, left_y, right_x, right_y = split
    return{
        "feature_index": feature_index,
        "threshold": threshold,
        "left": tree_builder(left_x, left_y, depth + 1, max_depth),
        "right": tree_builder(right_x, right_y, depth + 1, max_depth)
    }

#Decision Tree Function
def decision_tree(tree, row):
    if "label" in tree:
        return tree["label"]
    feature_index = tree["feature_index"]
    threshold = tree["threshold"]
    if row[feature_index] < threshold:
        return decision_tree(tree["left"], row)
    else:
        return decision_tree(tree["right"], row)

#Data to be used from Preprocessing(only numeric columns)
sampled_data = sampled_df.select_dtypes(include = ['number']).dropna()

#Features
numerized_features_all = numeric_features #all


#Split data into features and labels(all data)
x_all = sampled_data.values.tolist()
#Even though the dataset itself only has positive values for diabetes, we will randomize it so that we can see the deciison tree's output depending on the values of certain attributes
y_all = [1 if i < len(sampled_data) / 2 else 0 for i in range(len(sampled_data))] 

#Tree with all data set
all_tree = tree_builder(x_all, y_all)


#Testing for diabetes positivity with a varying inputs

#Returns a list of random nums based on the max and min of the attribute list

def rand_num(df, attributes):
    nums = []
    for attribute in attributes:
        max_val = df[attribute].max()
        min_val = df[attribute].min()
        num = random.randrange(min_val, max_val)
        nums.append(num)
    return nums


test_sample1 = x_all[0] #Example input from dataset
sample1_decision = decision_tree(all_tree, test_sample1)

test_sample2 = rand_num(sampled_df, numerized_features_all) #Custom input with random numbers
sample2_decision = decision_tree(all_tree, test_sample2)

test_sample3 = rand_num(sampled_df, numerized_features_all) #Custom input with random numbers
sample3_decision = decision_tree(all_tree, test_sample3)

test_sample4 = rand_num(sampled_df, numerized_features_all) #Custom input with random numbers
sample4_decision = decision_tree(all_tree, test_sample4)

test_sample5 = rand_num(sampled_df, numerized_features_all) #Custom input with random numbers
sample5_decision = decision_tree(all_tree, test_sample5)

print("\nCriteria: 1: Diabetes Positive, 0: Diabetes Negative")
print("List of tested attirbutes: ", numerized_features_all) 
print("Values for sample 1: ", test_sample1)
print("Sample-1 Prediction:", sample1_decision)
print("Values for sample 2: ", test_sample2)
print("Sample-2 Prediction:", sample2_decision)
print("Values for sample 3: ", test_sample3)
print("Sample-3 Prediction:", sample3_decision)
print("Values for sample 4: ", test_sample4)
print("Sample-4 Prediction:", sample4_decision)
print("Values for sample 5: ", test_sample5)
print("Sample-5 Prediction:", sample5_decision)

#Testing for diabetes positivity via set of symptomps, lifestyle choice, and hereditary traits


#for this make a custom x above specific for this problem so that 

#Features
specified_features = ['Family History', 'Dietary Habits', 'Early Onset Symptoms']

# #Split data into features and labels(specific data)

#Requires manual mapping for string type values
sampled_data_specified = sampled_df[specified_features].dropna()

def label_encoding(df):
    label_mapping = {}
    for column in df.columns[:-1]:  # Exclude the label column
        unique_values = list(set(df[column]))
        label_mapping[column] = {value: idx for idx, value in enumerate(unique_values)}
        df[column] = df[column].map(label_mapping[column])
    return df, label_mapping

# Encode the DataFrame
encoded_df, encoding_map = label_encoding(sampled_data_specified)

x_specified = encoded_df.values.tolist()
y_specified = [1 if i < len(encoded_df) / 2 else 0 for i in range(len(encoded_df))] #same explanation as above

# #Tree with specified data 
specified_tree = tree_builder(x_specified, y_specified)

test_sample1_specified = x_specified[0]
sample1_decision_specified = decision_tree(specified_tree, test_sample1_specified)

test_sample2_specified = [1, 0, 'No']
sample2_decision_specified = decision_tree(specified_tree, test_sample2_specified)

test_sample3_specified = [0, 0, 'No']
sample3_decision_specified = decision_tree(specified_tree, test_sample3_specified)

test_sample4_specified = [1, 1, 'No']
sample4_decision_specified = decision_tree(specified_tree, test_sample4_specified)

test_sample5_specified = [0, 1, 'Yes']
sample5_decision_specified = decision_tree(specified_tree, test_sample5_specified)

test_sample6_specified = [1, 0, 'Yes']
sample6_decision_specified = decision_tree(specified_tree, test_sample6_specified)

test_sample7_specified = [0, 0, 'Yes']
sample7_decision_specified = decision_tree(specified_tree, test_sample7_specified)

test_sample8_specified = [1, 1, 'Yes']
sample8_decision_specified = decision_tree(specified_tree, test_sample8_specified)


print("\nCriteria: 1: Diabetes Positive, 0: Diabetes Negative")
print("Attribute Mappings: ", encoding_map)
print("List of tested attirbutes: ", specified_features)
print("Values for specified sample 1: ", test_sample1_specified)
print("Specified-Sample-1 Prediction:", sample1_decision_specified)
print("Values for specified sample 2: ", test_sample2_specified)
print("Specified-Sample-2 Prediction:", sample2_decision_specified)
print("Values for specified sample 3: ", test_sample3_specified)
print("Specified-Sample-3 Prediction:", sample3_decision_specified)
print("Values for specified sample 4: ", test_sample4_specified)
print("Specified-Sample-4 Prediction:", sample4_decision_specified)
print("Values for specified sample 5: ", test_sample5_specified)
print("Specified-Sample-5 Prediction:", sample5_decision_specified)
print("Values for specified sample 6: ", test_sample6_specified)
print("Specified-Sample-6 Prediction:", sample6_decision_specified)
print("Values for specified sample 7: ", test_sample7_specified)
print("Specified-Sample-7 Prediction:", sample7_decision_specified)
print("Values for specified sample 8: ", test_sample8_specified)
print("Specified-Sample-8 Prediction:", sample8_decision_specified)


#K-Means Clustering

#Initializing first set of centroids
def init_centroids(data, k):
    if isinstance(data, dict):
        data = list(data.values())
    elif hasattr(data, 'to_numpy'):
        data = data.to_numpy().tolist()
    
    return random.sample(data,k)

#Calculating distance for cluster assignment
def euclidean_distance(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

#Cluster assignment to centroids
def cluster_assignment(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        nearest_index = distances.index(min(distances))
        clusters[nearest_index].append(point)
    return clusters

#Recalculating the centroids
def recalc_centroid(clusters, data=None, k=None):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroid = tuple(sum(dim) / len(cluster) for dim in zip(*cluster))
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(init_centroids(data,k)[0])
    return new_centroids

#The K-means algorithm
def kmeans(data, k, max_iter=200):
    centroids = init_centroids(data, k)

    for i in range(max_iter):
        clusters = cluster_assignment(data, centroids)
        new_centroids = recalc_centroid(clusters)

        if new_centroids == centroids:
            print("No new centroids. Finished.")
            break
        centroids = new_centroids

    return clusters, centroids


#Data to be used from Preprocessing(only numeric columns)
sampled_data = sampled_df.select_dtypes(include = ['number']).dropna()

#Features
numerized_features = numeric_features

#Can only be visualized in pairs of two
feature_set1 = [numeric_features[0], numeric_features[1]]
feature_set2 = [numeric_features[2], numeric_features[3]]
feature_set3 = [numeric_features[4], numeric_features[5]]
feature_set4 = [numeric_features[6], numeric_features[7]]
feature_set5 = [numeric_features[8], numeric_features[9]]
feature_set6 = [numeric_features[10], numeric_features[11]]
feature_set7 = [numeric_features[12]]

tupled_data1 = [tuple(row) for row in sampled_data[feature_set1].values]
tupled_data2 = [tuple(row) for row in sampled_data[feature_set2].values]
tupled_data3 = [tuple(row) for row in sampled_data[feature_set3].values]
tupled_data4 = [tuple(row) for row in sampled_data[feature_set4].values]
tupled_data5 = [tuple(row) for row in sampled_data[feature_set5].values]
tupled_data6 = [tuple(row) for row in sampled_data[feature_set6].values]
tupled_data7 = [tuple(row) for row in sampled_data[feature_set7].values]

#Number of clusters, 1 for each type of diabetes
k = len(sampled_df['Target'].unique())

clusters1, centroids1 = kmeans(tupled_data1, k)
clusters2, centroids2 = kmeans(tupled_data2, k)
clusters3, centroids3 = kmeans(tupled_data3, k)
clusters4, centroids4 = kmeans(tupled_data4, k)
clusters5, centroids5 = kmeans(tupled_data5, k)
clusters6, centroids6 = kmeans(tupled_data6, k)
clusters7, centroids7 = kmeans(tupled_data7, k)



#Visualization
colors = []
for i in range(len(numeric_features)):
    r = np.round(np.random.rand(), 1)
    g = np.round(np.random.rand(), 1)
    b = np.round(np.random.rand(), 1)
    colors.append((r,g,b))


plt.xlim(0,90)
plt.ylim(0,90)

#For feature set 1
for i, cluster in enumerate(clusters1):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids1)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()

#For feature set 2
for i, cluster in enumerate(clusters2):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids2)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()

#For feature set 3
for i, cluster in enumerate(clusters3):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids3)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()


#For feature set 4
for i, cluster in enumerate(clusters4):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids4)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()


#For feature set 5
for i, cluster in enumerate(clusters5):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids5)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()



#For feature set 6
for i, cluster in enumerate(clusters6):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids6)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()


#For feature set 7
for i, cluster in enumerate(clusters7):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i}')

x_centroid, y_centroid = zip(*centroids7)
plt.scatter(x_centroid, y_centroid, color = 'black', marker = 'X', s = 200, label = 'Centroids')

plt.title('K-means Clustering')
plt.legend()
plt.show()




#Apriori Algorithm







#Accuracy

#Precision

#F-1 Score

#Confusion Matrix

#AUC

#ROC