import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import classification_report, confusion_matrix, roc_auc_score
from model.all_names import (binary_cols, multi_cols, numeric_cols, passthrough_cols,
                             feature_cols, target_col, target_encoding)

RANDOM_STATE = 42 # A seed for randomizing - for reproducible results.
# (model + figures) saved in model_output/
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_output')
MODEL_PATH   = os.path.join(OUTPUT_DIR, 'churn_model.pkl')

# generate 'X' and 'y' - split df into input and output
def dataxy(data):
    X = data[feature_cols]
    y = data[target_col].map(target_encoding)
    return (X, y)

# categorical columns -> 0/1 columns, numerical and passthrough columns copied as they are.
def encode(X):
    out = pd.DataFrame(index=X.index)

    # binary: first value -> 0, second value -> 1
    for col, values in binary_cols.items():
        out[col] = (X[col] == values[1]).astype(int)

    # multi category: one 0/1 column per value, except the first value (the baseline - encoded into the rest)
    # adding new cols is essential because otherwise the model things that these are sequential.
    for col, values in multi_cols.items():
        for value in values[1:]:
            out[col + '_' + value] = (X[col] == value).astype(int)

    for col in numeric_cols:
        out[col] = X[col].astype(float)
    for col in passthrough_cols:
        out[col] = X[col]
    return out

# mean and standard deviation of each numerical column - learnt from the training rows only.
def getMeasures(X):
    scaler = {}
    for col in numeric_cols:
        scaler[col] = (float(X[col].mean()), float(X[col].std(ddof=0))) # not the default ddof=1.
    return scaler

# numerical columns -> (value - mean) / std, so they are on the same scale as the 0/1 columns.
def scale(X, scaler):
    X = X.copy()
    for col, (mean, std) in scaler.items():
        X[col] = (X[col] - mean) / std
    return X

def prepare(X, scaler):
    return scale(encode(X), scaler)


# P(churn) for raw customer rows, using a model returned by regress (or loaded from MODEL_PATH).
def predictProba(model, X):
    return model['reg'].predict_proba(prepare(X, model['scaler']))[:, 1]

def regress(data): # LogisticRegression
    # split test and train datasets: 80/20, stratified so both keep the same churn ratio
    X, y = dataxy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=RANDOM_STATE)

    # encode and scale each split once, and reuse the results below
    encoded_train  = encode(X_train)
    measures       = getMeasures(encoded_train)
    prepared_train = scale(encoded_train, measures)
    prepared_test  = prepare(X_test, measures)

    reg = LogisticRegression(class_weight='balanced', max_iter=1000)
    reg.fit(prepared_train, y_train)

    # everything needed to predict later - this dict is what gets saved
    model = {'reg': reg, 'scaler': measures}

    prediction  = reg.predict(prepared_test)
    probability = reg.predict_proba(prepared_test)[:, 1]

    a = classification_report(y_test, prediction, target_names=['No churn', 'Churn'])
    b =      confusion_matrix(y_test, prediction)

    return {'size': (X_train.shape[0], X_test.shape[0]), 'method': 'Logistic Regression', 'model': model,
            'classification_report': a, 'confusion_matrix': b, 'auc': roc_auc_score(y_test, probability),
            'X_test': X_test, 'y_test': y_test, 'y_prob': probability}


def saveModel(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path
