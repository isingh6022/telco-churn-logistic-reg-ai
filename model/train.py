import os
import matplotlib
matplotlib.use('Agg') # only save the plots, no popup windows
import matplotlib.pyplot as plt

from model import parser as parse
from model.models   import regress, saveModel, OUTPUT_DIR
from model.evaluate import printReport, plotConfusionMatrix, plotROC, oddsRatios

def main():
    # load and clean the data
    data = parse.cleanData(parse.loadData())
    print('\nshape of cleaned data : ', data.shape)
    print('\nClass distribution:-\n', parse.classDistribution(data), '\n')

    # train and evaluate
    results = regress(data)
    printReport(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, plot in [
            ('confusion_matrix.png', plotConfusionMatrix),
            ('roc_curve.png', plotROC)
        ]:
        fig = plot(results) # Figure created here - not in the list - so one by one.
        fig.savefig(os.path.join(OUTPUT_DIR, name))
        plt.close(fig)
    print('plots saved in : ', OUTPUT_DIR)

    # business interpretation
    drivers, retention = oddsRatios(results['model'])
    print('\nTop 5 drivers of churn:-\n', drivers.to_string())
    print('\nTop 5 retention factors:-\n', retention.to_string())

    print('\nmodel saved as : ', saveModel(results['model']))

if __name__ == '__main__':
    main()
