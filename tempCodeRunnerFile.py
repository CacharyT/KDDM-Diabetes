encoded_df, encoding_map = label_encoding(sampled_data_specified)

# x_specified = encoded_df.values.tolist()
# y_specified = [1 if i < len(encoded_df) / 2 else 0 for i in range(len(encoded_df))] #same explanation as above

# # #Tree with specified data 
# specified_tree = tree_builder(x_specified, y_specified)

# test_sample1_specified = x_specified[0]
# sample1_decision_specified = decision_tree(specified_tree, test_sample1_specified)

# test_sample2_specified = [1, 0, 'No']
# sample2_decision_specified = decision_tree(specified_tree, test_sample2_specified)

# test_sample3_specified = [0, 0, 'No']
# sample3_decision_specified = decision_tree(specified_tree, test_sample3_specified)

# test_sample4_specified = [1, 1, 'No']
# sample4_decision_specified = decision_tree(specified_tree, test_sample4_specified)

# test_sample5_specified = [0, 1, 'Yes']
# sample5_decision_specified = decision_tree(specified_tree, test_sample5_specified)

# test_sample6_specified = [1, 0, 'Yes']
# sample6_decision_specified = decision_tree(specified_tree, test_sample6_specified)

# test_sample7_specified = [0, 0, 'Yes']
# sample7_decision_specified = decision_tree(specified_tree, test_sample7_specified)

# test_sample8_specified = [1, 1, 'Yes']
# sample8_decision_specified = decision_tree(specified_tree, test_sample8_specified)


# print("\nCriteria: 1: Diabetes Positive, 0: Diabetes Negative")
# print("Attribute Mappings: ", encoding_map)
# print("List of tested attirbutes: ", specified_features)
# print("Values for specified sample 1: ", test_sample1_specified)
# print("Specified-Sample-1 Prediction:", sample1_decision_specified)
# print("Values for specified sample 2: ", test_sample2_specified)
# print("Specified-Sample-2 Prediction:", sample2_decision_specified)
# print("Values for specified sample 3: ", test_sample3_specified)
# print("Specified-Sample-3 Prediction:", sample3_decision_specified)
# print("Values for specified sample 4: ", test_sample4_specified)
# print("Specified-Sample-4 Prediction:", sample4_decision_specified)
# print("Values for specified sample 5: ", test_sample5_specified)
# print("Specified-Sample-5 Prediction:", sample5_decision_specified)
# print("Values for specified sample 6: ", test_sample6_specified)
# print("Specified-Sample-6 Prediction:", sample6_decision_specified)
# print("Values for specified sample 7: ", test_sample7_specified)
# print("Specified-Sample-7 Prediction:", sample7_decision_specified)
# print("Values for specified sample 8: ", test_sample8_specified)
# print("Specified-Sample-8 Prediction:", sample8_decision_specified)