#KDDM-Diabetes Dataset
#Contributors: Cachary Tolentino and Ian Valiante
#10/5/24

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import random
import math
from itertools import combinations, chain
from collections import defaultdict
from scipy.stats import mode

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

def plotOutliers(dfName):
    plt.boxplot(df[dfName])
    plt.title(f'{dfName} - Removed Outliers')
    plt.xlabel("Data")
    plt.ylabel("Values")
    plt.show()
    
plotOutliers('Waist Circumference')
plotOutliers('Pulmonary Function')






print("\n")






#Data Preprocessing (Sampling, Dimensionality Reduction, Feature Subset Selection, and Discretization)----------------------------------------------------------------------


#Sampling (with and without replacement; sample size = 10% of population)
print("Original population size: ", len(df))
sampled_df = df.sample(frac = 0.10, replace = True, random_state = 50) #keep random_state to ensure same values each time
print("New population size: ", len(sampled_df))

print("\n")


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
# sampled_df['Age'] = pd.cut(sampled_df['Age'], bins = [0, 12, 20, 60, sampled_df['Age'].max()], labels = ['Child', 'Teen', 'Adult', "Elderly"])
# print(sampled_df['Age'])

# sampled_df['BMI'] = pd.cut(sampled_df['BMI'], bins = [0, 18.5, 25, 30, sampled_df['BMI'].max()], right = False, labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Obese'])
# print(sampled_df['BMI'])





print("\n")







#Data Analayis & Implementation-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Confusion Matrix
def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix

#Accuracy
def calc_accuracy(conf_matrix):
    correct = np.trace(conf_matrix)  
    total = np.sum(conf_matrix)
    return correct / total

#Precision
def calc_precision(conf_matrix, num_classes):
    precision = []
    for cls in range(num_classes):
        tp = conf_matrix[cls][cls]
        fp = np.sum(conf_matrix[:, cls]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision.append(prec)
    return precision

#Recall
def calc_recall(conf_matrix, num_classes):
    recall = []
    for cls in range(num_classes):
        tp = conf_matrix[cls][cls]
        fn = np.sum(conf_matrix[cls, :]) - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall.append(rec)
    return recall

#F-1 Score
def calc_f1_score(precision, recall):
    f1_scores = []
    for prec, rec in zip(precision, recall):
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    return f1_scores

#Display Metrics (Precision, Recall, F1 Score)
def display_metrics(conf_matrix, target_mapping):
    num_classes = len(target_mapping)
    precision = calc_precision(conf_matrix, num_classes)
    recall = calc_recall(conf_matrix, num_classes)
    f1_scores = calc_f1_score(precision, recall)

    print("\nClass-wise Metrics:")
    for cls, label in enumerate(target_mapping.keys()):
        print(f"Class: {label}")
        print(f"  Precision: {precision[cls]:.2f}")
        print(f"  Recall: {recall[cls]:.2f}")
        print(f"  F1-score: {f1_scores[cls]:.2f}")

#Evaluation of Predictions (Confusion Matrix, Accuracy, Metrics)
def evaluate_predictions(y_true, y_pred, target_mapping):
    num_classes = len(target_mapping)
    conf_matrix = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = calc_accuracy(conf_matrix)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"\nAccuracy: {accuracy:.2f}")

    display_metrics(conf_matrix, target_mapping)



#Decision Tree----------------------------------------------------------------------------------------------------------------------------

#Leaf Node Constructor
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, class_distribution=None):
        self.feature = feature        # Feature for splitting
        self.threshold = threshold    # Threshold for splitting
        self.left = left              
        self.right = right            
        self.value = value             #Leaf class
        self.class_distribution = class_distribution  #needed for calculating AUC and ROC probabilities

#Splitting criteria (Gini Impurity)
def gini(data):
    count = np.bincount(data)
    probability = count / len(data)
    return 1 - sum(prob ** 2 for prob in probability)

#Splitting Effectiveness Criteria (Information Gain)
def information_gain(node, left_node, right_node):
    main_gini = gini(node)
    left_weight = len(left_node)/len(node)
    right_weight = len(right_node)/len(node)
    weighted_gini = (left_weight * gini(left_node) + right_weight * gini(right_node))
    return main_gini - weighted_gini

# Function to generate power set
def power_set(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

#Finding Best Split
def best_split(node1, node2, categorical_features):
    best_feature, best_threshold, best_gain = None, None, -1
    for feature in range (node1.shape[1]):

        if feature in categorical_features: #Handles categorical data
            unique_values = set(node1[:, feature])
            for subset in power_set(unique_values):  # Subsets for categorical
                if not subset or subset == unique_values:
                    continue
                left_mask = np.isin(node1[:, feature], subset)
                right_mask = ~left_mask
                left_node, right_node = node2[left_mask], node2[right_mask]
                if len(left_node) == 0 or len(right_node) == 0:
                    continue
                gain = information_gain(node2, left_node, right_node)
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, subset, gain

        else: #Handles numerical data
            thresholds = set(node1[:, feature])
            for threshold in thresholds:
                left_mask = node1[:, feature] <= threshold
                right_mask = ~left_mask
                left_node, right_node = node2[left_mask], node2[right_mask]
                if len(left_node) == 0 or len(right_node) == 0:               # No splits so skip
                    continue
                gain = information_gain(node2, left_node, right_node)
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain

    return best_feature, best_threshold

#Tree Builder
def build_tree(X, y, depth=0, max_depth=3, categorical_features = []):

    #Same labels so return
    if len(set(y)) == 1:
        class_distribution = {label: len(y) for label in set(y)} #For probabilitiy AUC and ROC
        return Node(value=y[0], class_distribution=class_distribution)
    
    #Max depth
    if depth >= max_depth:
        majority_class = max(set(y), key=list(y).count)
        class_distribution = {label: list(y).count(label) for label in set(y)}
        return Node(value=majority_class, class_distribution=class_distribution)
    
    #Best Split finder
    feature, threshold = best_split(X, y, categorical_features)
    if feature is None: 
        majority_class = max(set(y), key=list(y).count)
        return Node(value=majority_class)
    
    #Decision Splitting
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, categorical_features)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, categorical_features)
    
    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

#Predictor
def predict(sample, tree, return_probabilities=False):

    #Already leaf node so return (also returns probability)
    if tree.value is not None: 
        if return_probabilities and tree.class_distribution is not None:
            total_samples = sum(tree.class_distribution.values())
            probabilities = {cls: count / total_samples for cls, count in tree.class_distribution.items()}
            return probabilities
        return tree.value  # returns class label
    
    # Traverse through the tree via the thresholds
    if sample[tree.feature] <= tree.threshold:
        return predict(sample, tree.left, return_probabilities)
    else:
        return predict(sample, tree.right, return_probabilities)

#Predictor for all samples
def predict_all(X, tree, return_probabilities=False):
    return [predict(sample, tree, return_probabilities) for sample in X]


#Training the Tree----------------------------------------------------------------------------------------------------------------------------

#All features (except target)
all_features = sampled_df.columns.tolist()
all_features.remove('Target')

#Categorical features
categorical_features = sampled_df.select_dtypes(include = ['object']).columns.tolist()

#Values from all features
x = sampled_df[all_features].values


#Using Target (type of diabetes) feature as a classifier for prediction based on the other features
target_mapping = {label: idx for idx, label in enumerate(sampled_df['Target'].unique())}
y = sampled_df['Target'].map(target_mapping).values


#Using the dataset, split 95/5 for training and testing
split = int(0.95 * len(x))

#Training & testing Data
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

#Training the Tree
trained_tree_depth_3 = build_tree(x_train, y_train, categorical_features=categorical_features) #Depth 3, standard size
trained_tree_depth_5 = build_tree(x_train, y_train, max_depth = 5, categorical_features=categorical_features) #Depth 5, larger size


#Predictions----------------------------------------------------------------------------------------------------------------------------
diabetes_types = [key for key, value in target_mapping.items() if value in y_test]
print("Types of Diabetes:", diabetes_types)

print("\n")

#Prediction of depth 3
prediction_all_3 = [predict(sample, trained_tree_depth_3) for sample in x_test] #Predictions for all samples in the test set of depth 3

#Actual class labels for each prediction in prediction_all_3
predicted_3 = [diabetes_types[pred] for pred in prediction_all_3]
print("Predictions for depth 3:", predicted_3)

#Accuracy
accuracy_3 = np.mean(y_test == prediction_all_3)
print(f"Accuracy for depth 3: {accuracy_3:.2f}")

print("\n")

#Prediction of depth 5
prediction_all_5 = [predict(sample, trained_tree_depth_5) for sample in x_test] #Predictions for all samples in the test set of depth 5

#Actual class labels for each prediction in prediction_all_5
predicted_5 = [diabetes_types[pred] for pred in prediction_all_5]
print("Predictions for depth 5:", predicted_5)

#Accuracy
accuracy_5 = np.mean(y_test == prediction_all_5)
print(f"Accuracy for depth 5: {accuracy_5:.2f}")



#Evaluation----------------------------------------------------------------------------------------------------------------------------

print("\n")

#Probability of each class for each sample
probabilities_3 = [predict(sample, trained_tree_depth_3, return_probabilities=True) for sample in x_test]
probabilities_5 = [predict(sample, trained_tree_depth_5, return_probabilities=True) for sample in x_test]
num_of_classes = len(target_mapping)

print("\n")

#Evaluation of depth 3 (Confusion Matrix, Accuracy, Metrics)
print("Evaluation of Depth 3:")
evaluate_predictions(y_test, prediction_all_3, target_mapping)

print("\n")

#Evaluation of depth 5 (Confusion Matrix, Accuracy, Metrics)
print("Evaluation of Depth 5:")
evaluate_predictions(y_test, prediction_all_5, target_mapping)






print("\n")






#K-Means Clustering-----------------------------------------------------------------------

#Initializing first set of centroids
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

#Assigns the clusters
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

#Updates centroids based on assigned data
def update_centroids(data, clusters, k):
    centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return centroids

#Checks wether no more movement or very little for the centroids
def has_converged(old_centroids, new_centroids, tol=1e-4):
    return np.all(np.linalg.norm(old_centroids - new_centroids, axis=1) < tol)

# Calculate the most frequent diabetes type in each cluster
def assign_diabetes_type_to_clusters(clusters, labels_dict, data_labels, k):
    cluster_diabetes_types = []
    rev_labels = {v: k for k, v in labels_dict.items()}
    
    for i in range(k):
        cluster_mask = clusters == i
        cluster_labels = [data_labels[idx] for idx in range(len(data_labels)) if cluster_mask[idx]]
        
        if cluster_labels:
            series = pd.Series(cluster_labels)
            most_common = series.mode().iloc[0]  
        else:
            most_common = None  # Handle empty clusters
            
        cluster_diabetes_types.append(most_common)
    
    return cluster_diabetes_types

def k_means(data, labels, k, max_iterations=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    data_labels = sampled_df['Target'].values
    
    for iteration in range(max_iterations):

        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if has_converged(centroids, new_centroids, tol):
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        centroids = new_centroids

    cluster_diabetes_types = assign_diabetes_type_to_clusters(clusters, labels, data_labels, k)
    
    return centroids, clusters, cluster_diabetes_types


#Testing------------------------------------------------------------------------------------------------------------

#Data preparation

#Target column
diabetes_types = {label: idx for idx, label in enumerate(sampled_df['Target'].unique())}

#numerical features only (works best with K-Means) and only these to reduce dimensionality (less needed for cobminations)
numerical_features = ['Blood Glucose Levels', 'BMI', 'Age', 'Insulin Levels', 'Blood Pressure', 'Cholesterol Levels'] 

#All combinations of 3 features
feature_combinations = list(combinations(numerical_features, 3))

#Kmeans for each combination and graphing the results
for features in feature_combinations:
    data = sampled_df[list(features)].values
    centroids, clusters, cluster_diabetes_types = k_means(data, diabetes_types, 3)  # k = 3 clusters

    # Generate 3 plots for each combination of features (3 possible pairings)
    feature_pairs = [(0, 1), (1, 2), (0, 2)]  # Pairs of features to plot

    # Print cluster diabetes types
    print("\nCluster Diabetes Types for features:", features)
    for i, diabetes_type in enumerate(cluster_diabetes_types):
        print(f"Cluster {i}: {diabetes_type}")

    #Graph all 3 pairs of features with the clusters and centroids in one window
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"K-Means Clustering for Features: {features}")

    for ax, (x_idx, y_idx) in zip(axes, feature_pairs):
        ax.scatter(data[:, x_idx], data[:, y_idx], c=clusters, cmap='viridis')
        ax.scatter(centroids[:, x_idx], centroids[:, y_idx], c='red', marker='x', s=100)
        ax.set_xlabel(numerical_features[x_idx])
        ax.set_ylabel(numerical_features[y_idx])

    #Indicate the cluster diabetes types for each cluster and each graph
    for i, diabetes_type in enumerate(cluster_diabetes_types):
        for ax in axes:
            ax.text(centroids[i][feature_pairs[0][0]], centroids[i][feature_pairs[0][1]], diabetes_type, fontsize=12, color='red')


    plt.show()







print("\n")







#Apriori Algorithm-----------------------------------------------------------------------

#Data to be used from Preprocessing(only categorical features)
transaction_data = sampled_df.select_dtypes(include = ['object']).dropna()

#Reduce the number of features to 10 (dimenionaslity reduction for better performance)
transaction_data = transaction_data.sample(frac = 0.10, random_state = 50)


# Convert DataFrame to a list of transactions
def create_transactions(data):
    return data.apply(lambda row: set(row), axis=1).tolist()

transactions = create_transactions(transaction_data)

print("Current Transaction Data:", transactions)

print("\n")


#Frequent Itemset Generation
def generate_frequent_itemsets(transactions, min_support):
    itemsets = defaultdict(int)

    # Support count
    for transaction in transactions:
        for item in transaction:
            itemsets[frozenset([item])] += 1

    total_transactions = len(transactions)
    frequent_itemsets = {
        itemset: count / total_transactions
        for itemset, count in itemsets.items()
        if count / total_transactions >= min_support
    } #Initial frequent itemsets of size 1

    # Loop till no more frequent itemsets are found
    k = 2
    while True:

        #Generate candidate itemsets
        candidates = defaultdict(int)
        for itemset in frequent_itemsets:
            for other_itemset in frequent_itemsets:
                candidate = itemset.union(other_itemset)
                if len(candidate) == k:
                    if all(subset in frequent_itemsets for subset in combinations(candidate, k-1)): #Pruning infrequent subsets
                        candidates[candidate] += 1

        if not candidates:
            break

        #Prunes candidate itemsets that do not meet the minimum support
        candidate_supports = {
            itemset: count / total_transactions
            for itemset, count in candidates.items()
            if count / total_transactions >= min_support
        }

        # If no frequent itemsets were found for this k, break out of the loop
        if not candidate_supports:
            break

        frequent_itemsets = candidate_supports
        k += 1

    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []

    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for prev_size in range(1, len(itemset)):
                for prev in combinations(itemset, prev_size):
                    prev = frozenset(prev)
                    diff = itemset.difference(prev)

                    # Compute confidence of the rule
                    prev_support = frequent_itemsets.get(prev, 0)
                    if prev_support > 0:
                        rule_confidence = support / prev_support
                        if rule_confidence >= min_confidence:
                            rules.append((prev, diff, rule_confidence))

                    if rule_confidence >= min_confidence:
                        rules.append((prev, diff, rule_confidence))

    return rules

def apriori(transactions, min_support, min_confidence):
    frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
    print("\nFrequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"{set(itemset)}: {support:.2f}")

    association_rules = generate_association_rules(frequent_itemsets, min_confidence)
    print("\nAssociation Rules:")
    for rule in association_rules:
        antecedent, consequent, confidence = rule
        print(f"{set(antecedent)} => {set(consequent)}: {confidence:.2f}")


#Testing
apriori(transactions, min_support=0.001, min_confidence=0.5)
apriori(transactions, min_support=0.001, min_confidence=0.0005)