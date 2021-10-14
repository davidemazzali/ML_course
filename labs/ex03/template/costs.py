# -*- coding: utf-8 -*-

"""Functions used to compute the loss."""

import numpy as np


# computes MSE loss function
def compute_loss_mse(y, tx, w):
    e = y-tx.dot(w)
    return 1/(2*len(y)) * np.sum(e*e)

# computes MAE loss function
def compute_loss_mae(y, tx, w):
    e = y-tx.dot(w)
    return np.sum(np.abs(e))*1/(2*len(y))

def compute_loss_rmse(y, tx, w):
    return np.sqrt(2*compute_loss_mse(y, tx, w))