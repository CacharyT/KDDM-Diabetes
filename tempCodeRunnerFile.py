predictions_5 = [key for key, value in target_mapping.items() if value == prediction_all_5]
if predictions_5:
    predictions_5 = predictions_5[0]
else:
    predictions_5 = None
