# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]

# 2D grid search for MSE objective
def grid_search_mse(y, tx, w0, w1):
    losses = np.zeros((len(w0), len(w1)))
    
    for i, z0 in enumerate(w0):
        for j, z1 in enumerate(w1):
            losses[i,j] = costs.compute_loss_mse(y,tx,np.array([z0,z1]))
    
    return losses

# 2D grid search for MAE objective
def grid_search_mae(y, tx, w0, w1):
    losses = np.zeros((len(w0), len(w1)))
    
    for i, z0 in enumerate(w0):
        for j, z1 in enumerate(w1):
            losses[i,j] = costs.compute_loss_mae(y,tx,np.array([z0,z1]))
    
    return losses