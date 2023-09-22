import xgboost as xgb

import pandas as pd

from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from horse_data import load_and_preprocess

def xgb_hyperparameter_search(job_id):

    X, y, test, test_ids = load_and_preprocess()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    xgb_train_data = xgb.DMatrix(data=X, label=y_encoded)

    # Set XGBoost parameters
    param = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': 3,  # Number of classes
        'booster': 'gbtree',
        'eval_metric': 'mlogloss'
    }

    # Parameter grid
    param_dist = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'min_child_weight': [1, 2, 3, 4],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'n_estimators': [100, 200, 500, 1000, 2000],
        'objective': ['multi:softmax'],
        'num_class': [3]
    }

    clf = xgb.XGBClassifier()

    # Initialize RandomizedSearchCV
    rs = RandomizedSearchCV(clf,
                            param_distributions=param_dist,
                            n_iter=2000,
                            scoring='accuracy',
                            n_jobs=-1,
                            cv=5,
                            verbose=4,
                            random_state=42)

    print(f'Starting search, choosing {rs.n_iter} samples from a maximum of {len(ParameterGrid(param_dist))} combinations.')

    # Fit
    rs.fit(X, y_encoded)

    print(rs.best_params_, rs.best_score_)

    # Save the model
    rs.best_estimator_.save_model(f'xgb_model{job_id}.json')

    bst = rs.best_estimator_

    test_predictions = bst.predict(test)

    test_predictions_labels = label_encoder.inverse_transform(test_predictions.astype(int))

    submission = pd.DataFrame({
        'id': test_ids,  # Assuming test_data is your original test dataset
        'outcome': test_predictions_labels
    })

    submission.to_csv('submission_xgb2.csv', index=False)
