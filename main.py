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
from itertools import combinations, chain
from collections import defaultdict

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
# sampled_df['Age'] = pd.cut(sampled_df['Age'], bins = [0, 12, 20, 60, sampled_df['Age'].max()], labels = ['Child', 'Teen', 'Adult', "Elderly"])
# print(sampled_df['Age'])

# sampled_df['BMI'] = pd.cut(sampled_df['BMI'], bins = [0, 18.5, 25, 30, sampled_df['BMI'].max()], right = False, labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Obese'])
# print(sampled_df['BMI'])


#Data Analayis & Implementation-----------------------------------------------------------------------------------------------------------------------



#Decision Tree-----------------------------------------------------------------


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

#Finding Best Split
def best_split(node1, node2):
    best_feature, best_threshold, best_gain = None, None, -1
    for feature in range (node1.shape[1]):
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
def build_tree(X, y, depth=0, max_depth=3):

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
    feature, threshold = best_split(X, y)
    if feature is None: 
        majority_class = max(set(y), key=list(y).count)
        return Node(value=majority_class)
    
    #Decision Splitting
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    
    #Create the node
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


def predict_all(X, tree, return_probabilities=False):
    return [predict(sample, tree, return_probabilities) for sample in X]



#Training

#Numeric Features
numerized_features = numeric_features 

#Values from features
x = sampled_df[numerized_features].values


#Using Target (type of diabetes) feature as a classifier for prediction based on numerical features
target_mapping = {label: idx for idx, label in enumerate(sampled_df["Target"].unique())}
y = sampled_df["Target"].map(target_mapping).values


#Using the dataset, split 80/20 for training and testing
split = int(0.8 * len(x))

#Training & testing Data
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

#Training the Tree
trained_tree = build_tree(x_train, y_train, max_depth = 3) #arbitrary depth, may affect results, lower # = less complex


#Overall prediction w/ own dataset
predictions = [predict(row, trained_tree) for row in x]


#Custom predictions with pre-defined data

#Pre-defined "normal" levels
normal_levels = [
    24,   # Insulin Levels (under 25)
    38,   # Age (average age in America - 2022)
    20,   # BMI (between 18.5 and 24.9)
    80,   # Blood Pressure (random)
    150,  # Cholesterol Levels (<200)
    99,   # Waist Circumference (average in cm)
    80,   # Blood Glucose Levels (between 70 and 100)
    30,   # Weight Gain During Pregnancy (between 25 nd 35)
    90,   # Pancreatic Health (random)
    95,   # Pulmonary Function (random)
    88,   # Neurological Assessments (random)
    50,   # Digestive Enzyme Levels (random)
    3.5   # Birth Weight (between 5.5 to 10 lb but in kg)
]

high_Insulin = [
    80,    
    38,   
    20,   
    80,   
    150,  
    99,   
    80,   
    30,   
    90,   
    95,   
    88,   
    50,   
    3.5   
]

high_BMI = [
    24,    
    38,   
    30,   
    80,   
    150,  
    99,   
    80,   
    30,   
    90,   
    95,   
    88,   
    50,   
    3.5   
]

high_Cholesterol = [
    24,    
    38,   
    20,   
    80,   
    300,  
    99,   
    80,   
    30,   
    90,   
    95,   
    88,   
    50,   
    3.5   
]

high_Glucose = [
    24,    
    38,   
    20,   
    80,   
    150,  
    99,   
    150,   
    30,   
    90,   
    95,   
    88,   
    50,   
    3.5   
]


high_Insulin = np.array(high_Insulin)
high_BMI = np.array(high_BMI)
high_Cholesterol = np.array(high_Cholesterol)
high_Glucose = np.array(high_Glucose)

#Predictions
p_Insulin = predict(high_Insulin, trained_tree)
p_BMI = predict(high_BMI, trained_tree)
p_Cholesterol = predict(high_Cholesterol, trained_tree)
p_Glucose = predict(high_Glucose, trained_tree)

# Map the numeric label back to the class name
predicted_c1 = [key for key, value in target_mapping.items() if value == p_Insulin][0]
predicted_c2 = [key for key, value in target_mapping.items() if value == p_BMI][0]
predicted_c3 = [key for key, value in target_mapping.items() if value == p_Cholesterol][0]
predicted_c4 = [key for key, value in target_mapping.items() if value == p_Glucose][0]

print("\nPredicted class for the high insluin:", predicted_c1)
print("Predicted class for the high BMI:", predicted_c2)
print("Predicted class for the high cholesterol:", predicted_c3)
print("Predicted class for the high glucose:", predicted_c4)









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


def k_means(data, k, max_iterations=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iterations):

        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if has_converged(centroids, new_centroids, tol):
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        centroids = new_centroids
    
    return centroids, clusters


#Testing-------------------------------------------------------------------------

#Data prep
numerical_features = numeric_features
clustering_data = sampled_df[numerical_features].values


#Visualization specifically for Target and Symptoms

#Dataset
cluster_sample = sampled_df

#Encoding Target and early onset sympstoms
target_map = target_mapping
early_onset_map = {'Yes': 1, 'No': 0}
symptom_features = ['Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels', 'Blood Glucose Levels']

cluster_sample['Target_encoded'] = cluster_sample['Target'].map(target_map)
cluster_sample['early_onset_symptoms_encoded'] = cluster_sample['Early Onset Symptoms'].map(early_onset_map)

features = ['Target_encoded', 'early_onset_symptoms_encoded'] + symptom_features
data = cluster_sample[features].values

centroids_s, clusters_s = k_means(data, 3)

# Plotting the results
plt.figure(figsize=(8, 6))

# Scatter plot of the data points, colored by clusters
scatter = plt.scatter(cluster_sample['Target_encoded'], cluster_sample['early_onset_symptoms_encoded'], c=clusters_s, cmap='viridis', marker='o', alpha=0.6)

# Plot the centroids in red with a larger 'X' marker
plt.scatter(centroids_s[:, 0], centroids_s[:, 1], c='red', marker='X', s=200, label='Centroids')

# Adding labels and title
plt.title("K-Means Clustering with Target and Early Onset Symptoms")
plt.xlabel('Target')
plt.ylabel('Early Onset Symptoms')
plt.legend()
plt.colorbar(scatter, label='Cluster ID')
plt.show()


#Visualization specifically for Target and Insulin Level
# Plotting the results
plt.figure(figsize=(8, 6))

# Scatter plot of the data points, colored by clusters
scatter = plt.scatter(cluster_sample['Target_encoded'], cluster_sample['Insulin Levels'], c=clusters_s, cmap='viridis', marker='o', alpha=0.6)

# Plot the centroids in red with a larger 'X' marker
plt.scatter(centroids_s[:, 0], centroids_s[:, 1], c='red', marker='X', s=200, label='Centroids')

# Adding labels and title
plt.title("K-Means Clustering with Target and Insulin Levels")
plt.xlabel('Target')
plt.ylabel('Insulin Levels')
plt.legend()
plt.colorbar(scatter, label='Cluster ID')
plt.show()








#Apriori Algorithm

#Data to be used from Preprocessing(only categorical features)
transaction_data = sampled_df.select_dtypes(include = ['object']).dropna()

# Convert DataFrame to a list of transactions
def create_transactions(data):
    return data.apply(lambda row: set(row), axis=1).tolist()

transactions = create_transactions(transaction_data)


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
    }

    # Generate itemsets with higher k
    k = 2
    while True:

        #Generate candidate itemsets
        candidates = defaultdict(int)
        for itemset in list(frequent_itemsets.keys()):
            for other_itemset in frequent_itemsets.keys():
                if len(itemset.union(other_itemset)) == k:
                    candidates[itemset.union(other_itemset)] += 1

        if not candidates:
            break

        # Check support for new itemsets
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
                    rule_confidence = support / prev_support

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
apriori(transactions, min_support=0.001, min_confidence=0.001)








#Confusion Matrix, Accuracy, Precision, Recall, F-1 Score-----------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix

def calc_accuracy(conf_matrix):
    correct = np.trace(conf_matrix)  
    total = np.sum(conf_matrix)
    return correct / total

def calc_precision(conf_matrix, num_classes):
    precision = []
    for cls in range(num_classes):
        tp = conf_matrix[cls][cls]
        fp = np.sum(conf_matrix[:, cls]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision.append(prec)
    return precision

def calc_recall(conf_matrix, num_classes):
    recall = []
    for cls in range(num_classes):
        tp = conf_matrix[cls][cls]
        fn = np.sum(conf_matrix[cls, :]) - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall.append(rec)
    return recall

def calc_f1_score(precision, recall):
    f1_scores = []
    for prec, rec in zip(precision, recall):
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    return f1_scores

def mean(metrics):
    return np.mean(metrics)

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

    mean_precision = mean(precision)
    mean_recall = mean(recall)
    mean_f1 = mean(f1_scores)

    print("\nMacro-Average Metrics:")
    print(f"  Precision: {mean_precision:.2f}")
    print(f"  Recall: {mean_recall:.2f}")
    print(f"  F1-score: {mean_f1:.2f}")

def evaluate_predictions(y_true, y_pred, target_mapping):
    num_classes = len(target_mapping)
    conf_matrix = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = calc_accuracy(conf_matrix)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"\nAccuracy: {accuracy:.2f}")

    display_metrics(conf_matrix, target_mapping)



#Evaluating Models (if applicable)

#Decision Tree Evaluation
dt_conf_matrix = confusion_matrix(y_test, predictions, len(target_mapping))
display_metrics(dt_conf_matrix, target_mapping)

#K-Means Clustering Evaluation
#NO NEED FOR CLUSTERING SINCE IT IS NOT A CLASSIFICATION PROBLEM
#NO NEED FOR CONFUSION MATRIX BECAUSE ITS NOT PREDICTING ANY CLASS LABELS

#Apriori Evaluation
#NO NEED ALSO BECAUSE ITS AN ASSOCIATION BASED ALGORITHM NOT FOR CLASSIFICATION
#NO NEED FOR CONFUSION MATRIX BECAUSE ITS NOT PREDICTING ANY CLASS LABELS



#AUC and ROC-------------------------------------------------------------------------------------------------------------------------
def calculate_roc_auc(y_true, y_pred_probs):
    thresholds = np.linspace(0, 1, 100)  # Generate thresholds from 0 to 1
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate

    for threshold in thresholds:
        # Convert probabilities to binary predictions at the threshold
        y_pred = (y_pred_probs >= threshold).astype(int)

        # Calculate TP, FP, FN, TN
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))

        # Calculate TPR and FPR
        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)

    # Calculate AUC using the trapezoidal rule
    auc = np.trapz(tpr, fpr)  # Integrate TPR vs. FPR

    return tpr, fpr, auc


#Visualization for AUC and ROC (if applicable)

#Decision tree AUC and ROC
# Predicting probabilities for all samples in x_test
predictions_probabilities = predict_all(x_test, trained_tree, return_probabilities=True)

# If you want the probabilities for a particular class (e.g., positive class with index 1)
positive_class_index = 1 # 1 because target is type of diabetes which are all diabetes positive
y_pred_probs = [prob.get(positive_class_index, 0) for prob in predictions_probabilities]
tpr, fpr, auc = calculate_roc_auc(y_test, y_pred_probs)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()