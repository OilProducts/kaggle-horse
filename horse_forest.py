from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid

import pickle

import horse_data

def forest_hyperparam_search(job_id):
    # Hyperparameter grid for Random Forest
    # param_dist = {
    #     'bootstrap': [True, False],
    #     'max_depth': [10, 20, 30, 40, 50, None],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'min_samples_leaf': [1, 2, 4, 6, 8],
    #     'min_samples_split': [2, 5, 10, 15, 20],
    #     'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000]
    # }


    # Load and preprocess data
    X, y, test, test_ids = horse_data.load_and_preprocess()

    param_dist = {
        'bootstrap': [True, False],
        'max_depth': [5, 7, 9, 10, 20, 30, 40, 50, 60, 70, None],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 3, 5, 7, 10, 12, 15, 20],
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }

    # Initialize RandomForest and RandomizedSearchCV
    rf = RandomForestClassifier(random_state=42)

    # Using 5-fold cross-validation and 100 iterations
    rf_random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                          n_iter=1000, cv=5, verbose=1,
                                          random_state=42, n_jobs=-1,
                                          scoring='accuracy')

    print(
        f'Starting search, choosing {rf_random_search.n_iter} samples from a maximum of {len(ParameterGrid(param_dist))} combinations.')

    # Fit the random search model
    rf_random_search.fit(X, y)

    print(f'Best score obtained: {rf_random_search.best_score_} for {rf_random_search.best_params_}')

    # Save the best model
    with open(f'./models/rf_model_{job_id}.pkl', 'wb') as f:
        pickle.dump(rf_random_search.best_estimator_, f)

    return rf_random_search.best_params_