import pandas as pd

from sklearn.preprocessing import StandardScaler


def load_and_preprocess():
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv("./test.csv")
    horse_data = pd.read_csv("./horse.csv")

    train_data.drop(columns=['id'], inplace=True)
    train_data = pd.concat([train_data, horse_data], axis=0, ignore_index=True)

    # Store the 'id' column in a separate variable
    test_ids = test_data['id']

    # Drop irrelevant columns
    train_data.drop(columns=['hospital_number'], inplace=True)
    test_data.drop(columns=['id', 'hospital_number'], inplace=True)

    missing_values = train_data.isnull().sum()
    columns_with_missing_values = missing_values[missing_values > 0]

    numerical_columns_with_missing_values = [col for col in columns_with_missing_values.index if
                                             train_data[col].dtype != 'object']
    categorical_columns_with_missing_values = [col for col in columns_with_missing_values.index if
                                               train_data[col].dtype == 'object']

    # Numerical columns imputation
    for col in numerical_columns_with_missing_values:
        median_value = train_data[col].median()
        train_data[col].fillna(median_value, inplace=True)

    # Categorical columns imputation
    for col in categorical_columns_with_missing_values:
        mode_value = train_data[col].mode()[0]
        train_data[col].fillna(mode_value, inplace=True)

    # Identify numerical columns to scale
    numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

    # Scale numerical features
    scaler = StandardScaler()
    train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
    test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

    # Split data into features and target
    X = train_data.drop(columns=['outcome'])
    y = train_data['outcome']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)
    test_encoded = pd.get_dummies(test_data)

    # Ensure the test set has the same columns as the training set (after encoding),
    # this can happen if the test set did not have an example of a specific value
    # when one hot encoding the data. Fill missing columns with 0.
    for col in X_encoded.columns:
        if col not in test_encoded.columns:
            test_encoded[col] = 0

    # Ensure the training set has the same columns as the test set (after encoding),
    # this can happen if the training set did not have an example of a specific value
    # when one hot encoding the data. Fill missing columns with 0.
    for col in test_encoded.columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    # Ensure the order of columns in the training dataset matches the test dataset
    X_encoded = X_encoded[test_encoded.columns]

    return X_encoded, y, test_encoded, test_ids
