import os
import pandas as pd
from model.all_names import data_file, target_col, numeric_cols

# the csv lives in the project root, one level above this package
def loadData(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_file)
    return pd.read_csv(path)

def cleanData(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    blanks = df['TotalCharges'].isna()
    # imputing 0 is only safe if all the blanks are new customers - stop otherwise.
    assert (df.loc[blanks, 'tenure'] == 0).all(), 'blank TotalCharges found for tenure > 0'
    df.loc[blanks, 'TotalCharges'] = 0.0

    print('blank TotalCharges imputed with 0 : %d rows (all with tenure = 0)' % blanks.sum())
    print('null values left after cleaning  : %d' % df.isna().sum().sum())

    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def classDistribution(df):
    counts = df[target_col].value_counts()

    return pd.DataFrame({'count': counts, 'percent': (100 * counts / counts.sum()).round(2)})
