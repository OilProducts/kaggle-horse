import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import horse_data



best_hyperparameters = {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2',
                        'max_depth': 30, 'criterion': 'entropy', 'class_weight': 'balanced', 'bootstrap': False}


# Load and preprocess data
X, y, test, test_ids = load_and_preprocess()

# Initialize and train the RandomForest model using the best hyperparameters
best_rf_model = RandomForestClassifier(**best_hyperparameters, random_state=42)
best_rf_model.fit(X, y)  # Train on the entire training dataset

# Generate predictions on the test set
test_predictions = best_rf_model.predict(test)
print(test_predictions[:10])

# unique_classes = train_data['outcome'].unique()
# label_mapping = {index: label for index, label in enumerate(unique_classes)}
# predictions_labels = [label_mapping[pred] for pred in test_predictions]

# print(test_data)
# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'outcome': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)



