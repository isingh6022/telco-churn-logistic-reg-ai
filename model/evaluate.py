import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

def printReport(results):
    print('# %s\n' % results['method'])
    print('Size of training sample = %d\nSize of testing  sample = %d\n' % (
        results['size'][0], results['size'][1]))
    print('ROC AUC = %0.3f\n' % results['auc'])
    print('Classification Report:-\n', results['classification_report'])
    print('Confusion Matrix:-\n\n', results['confusion_matrix'])
    print('_________________________________________________________\n')

def plotConfusionMatrix(results):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ConfusionMatrixDisplay(results['confusion_matrix'], display_labels=['No churn', 'Churn']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix - %s' % results['method'])
    fig.tight_layout()
    return fig

def plotROC(results):
    model_fpr, model_tpr, _ = roc_curve(results['y_test'], results['y_prob'])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.plot(model_fpr, model_tpr, linestyle='-', label='%s (auc = %0.3f)' % (results['method'], results['auc']))
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random (auc = 0.500)')
    ax.set_xlabel('False positive rate -->')
    ax.set_ylabel('True positive rate -->')
    ax.set_title('ROC curve')
    ax.legend()
    fig.tight_layout()
    return fig

# Coefficients -> odds ratios (exp(coef)).
# TODO: Look into how odds ratios work.
def oddsRatios(model, n=5):
    names = model['reg'].feature_names_in_   # the encoded column names, in the order fitted
    coefs = model['reg'].coef_[0]

    df = pd.DataFrame({'feature': names, 'coefficient': coefs, 'odds_ratio': np.exp(coefs)})
    df['change_in_odds'] = ['%+.0f%%' % (100 * (ratio - 1)) for ratio in df['odds_ratio']]
    df = df.sort_values('coefficient', ascending=False).round(3).reset_index(drop=True)

    drivers   = df.head(n).reset_index(drop=True)                     # increase churn
    retention = df.tail(n).iloc[::-1].reset_index(drop=True)          # decrease churn, strongest first
    drivers.index   += 1
    retention.index += 1
    return drivers, retention
