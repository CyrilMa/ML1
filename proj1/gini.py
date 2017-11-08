#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation """

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)



def gini_visualization(actual, pred, graph = False):
    gini_predictions = gini(actual, pred)
    gini_max = gini(actual, actual)
    ngini= gini_normalized(actual, pred)
    print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))
    
    
    if(graph):
        # Sort the actual values by the predictions
        data = zip(actual, pred)
        sorted_data = sorted(data, key=lambda d: d[1])
        sorted_actual = [d[0] for d in sorted_data]
        # print('Sorted Actual Values', sorted_actual)
        
        # Sum up the actual values
        cumulative_actual = np.cumsum(sorted_actual)
        cumulative_index = np.arange(1, len(cumulative_actual)+1)
        
        plt.plot(cumulative_index, cumulative_actual)
        plt.xlabel('Cumulative Number of Predictions')
        plt.ylabel('Cumulative Actual Values')
        plt.show()
    return(ngini)

    
    
