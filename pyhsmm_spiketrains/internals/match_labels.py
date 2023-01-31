#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:24:22 2022

@author: jonat
"""

import numpy as np
from pyhsmm_spiketrains.internals.hungarian import linear_sum_assignment


def get_overlap_mat(x, y, method="cosine_sim"):
    # With normalization, the overlap as the number of matching instances is
    # interpretable as cosine similarity, facilitating conversion to cosine
    # distance when forming the cost matrix.

    x = np.squeeze(x)
    y = np.squeeze(y)

    # Get state labels.
    states_x = np.unique(x)
    states_y = np.unique(y)

    # Preallocate.
    overlap = np.empty((len(states_x), len(states_y)))

    # Calculate elements of overlap matrix as cosine similarities.
    for i_x, state_x in enumerate(states_x):
        for i_y, state_y in enumerate(states_y):
            overlap[i_x, i_y] = calc_overlap(x == state_x,
                                             y == state_y,
                                             method=method)

    return overlap


def calc_overlap(x, y, method="cosine_sim"):
    if method == "cosine_sim":
        overlap = cosine_sim(x.astype("float"), y.astype("float"))
    elif method == "count":
        overlap = np.sum(x & y)

    return overlap


def cosine_sim(a, b):
    cs = np.sum((a / np.linalg.norm(a)) * (b / np.linalg.norm(b)))
    return cs


def convert_overlap_to_cost(overlap, method="cosine_sim", n_obs=[]):
    if method == "cosine_sim":
        cost = 1 - overlap
    elif method == "count":
        assert not n_obs, "If method is 'cosine_sim', n_obs must be passed in " + \
                          "as the number of total observations."
        cost = n_obs - overlap
    return cost


def get_optimal_assignment(x, y, method="cosine_sim"):
    assert len(x) == len(y), "x and y must be of the same length."

    overlap = get_overlap_mat(x, y, method=method)
    cost = convert_overlap_to_cost(overlap, method=method, n_obs=len(x))

    row_ind, col_ind = linear_sum_assignment(cost)

    return row_ind, col_ind, overlap, cost

