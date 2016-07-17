# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:50:18 2016

@author: Nicholas Sherer
"""

import numpy as np
from functools import partial


def sumMeans(x, m, y, n):
    """
    To subtract means just make n negative.
    """
    return (m*x+n*y)/(m+n), m+n


def varFromMeans(mean_sqr, mean):
    return mean_sqr - mean**2


def biThresh2(array):
    ordr = np.sort(array, axis=None).flatten()
    ordr_sqr = ordr**2
    n = np.size(array)
    mcounts = np.arange(1, n+1, 1)
    ncounts = n + 1 - mcounts
    ls, lm, hs, hm = cumSumnMeans(ordr, mcounts, ncounts)
    lsq, lmsq, hsq, hmsq = cumSumnMeans(ordr_sqr, mcounts, ncounts)
    low_var = lmsq - lm**2
    high_var = hmsq - hm**2
    total_var = low_var*mcounts+high_var*ncounts
    index = np.argmin(total_var)
    thresh = ordr[index]
    return total_var, thresh


def triThresh2(array, step_size):
    ordr = np.sort(array, axis=None).flatten()
    ordr_sqr = ordr**2
    n = np.size(array)
    mcounts = np.arange(1, n+1, 1)
    ncounts = n + 1 - mcounts
    ls, lm, hs, hm = cumSumnMeans(ordr, mcounts, ncounts)
    lsq, lmsq, hsq, hmsq = cumSumnMeans(ordr_sqr, mcounts, ncounts)
    low_var = lmsq - lm**2
    high_var = hmsq - hm**2
    mid_mean = partial(midMeanSortArr, mu=np.sum(ordr)/n, p=n, sort1=lm,
                       sort2=hm)
    mid_meansqr = partial(midMeanSortArr, mu=np.sum(ordr_sqr)/n, p=n,
                          sort1=lmsq, sort2=hmsq)
    total_var = 10**23
    low_thr = 0
    high_thr = 0
    for i in range(1, n, step_size):
        for j in range(1, i, step_size):
            mm = mid_mean(l=j, m=n-i)
            mmsq = mid_meansqr(l=j, m=n-i)
            mid_var = mmsq - mm**2
            total_var_new = j*low_var[j]+(i-j)*mid_var+(n-i)*high_var[n-i]
            if total_var_new < total_var:
                total_var = total_var_new
                low_thr = j
                high_thr = i
    return ordr[low_thr], ordr[high_thr], total_var


def cumSumnMeans(ordr, mcounts, ncounts):
    low_sums = np.cumsum(ordr)
    low_means = low_sums / mcounts
    high_sums = low_sums[-1] - low_sums
    high_means = high_sums / ncounts
    return low_sums, low_means, high_sums, high_means


def midMean(mu, p, x, l, y, m):
    return (p*mu - l*x - m*y)/(p-l-m)


def midMeanSortArr(mu, p, sort1, l, sort2, m):
    x = sort1[l]
    y = sort2[p-m]
    return (p*mu - l*x - m*y)/(p-l-m)


def biThresh(array):
    ordr = np.sort(array, axis=None)
    ordr_sqr = np.square(ordr)
    n_total = np.size(ordr)
    # maxi = np.max(ordr)
    # mini = np.min(ordr)
    cl1_mean = np.mean(ordr)
    cl1_mean_sqr = np.mean(ordr_sqr)
    cl1_var = varFromMeans(cl1_mean_sqr, cl1_mean)
    cl1_n = n_total
    cl2_mean = 0
    cl2_mean_sqr = 0
    cl2_var = 0
    cl2_n = 0
    tot_var = cl1_n*cl1_var + cl2_n*cl2_var
    thr = 0
    # tot_var_list = [tot_var]
    for i in range(0, n_total):
        n_thr = ordr[i]
        n_cl1_mean, n_cl1_n = delItemMean(cl1_mean, cl1_n, n_thr, 1)
        n_cl1_mean_sqr, n_cl1_n = delItemMean(cl1_mean_sqr, cl1_n, n_thr**2, 1)
        n_cl1_var = varFromMeans(n_cl1_mean_sqr, n_cl1_mean)
        n_cl2_mean, n_cl2_n = addItemMean(cl2_mean, cl2_n, n_thr, 1)
        n_cl2_mean_sqr, n_cl2_n = addItemMean(cl2_mean_sqr, cl2_n, n_thr**2, 1)
        n_cl2_var = varFromMeans(n_cl2_mean_sqr, n_cl2_mean)
        n_tot_var = n_cl1_var * n_cl1_n + n_cl2_var * n_cl2_n
        # tot_var_list.append(n_tot_var)
        if n_tot_var > tot_var:
            tot_var = n_tot_var
            thr = n_thr
        cl1_mean = n_cl1_mean
        cl1_mean_sqr = n_cl1_mean_sqr
        cl1_var = n_cl1_var
        cl1_n = n_cl1_n
        cl2_mean = n_cl2_mean
        cl2_mean_sqr = n_cl2_mean_sqr
        cl2_var = n_cl2_var
        cl2_n = n_cl2_n
    return thr, tot_var


def triThresh(array):
    ordr = np.sort(array, axis=None)
    ordr_sqr = np.square(ordr)
    n_total = np.size(ordr)
    # maxi = np.max(ordr)
    # mini = np.min(ordr)
    cl1_mean = np.mean(ordr)
    cl1_mean_sqr = np.mean(ordr_sqr)
    cl1_var = varFromMeans(cl1_mean_sqr, cl1_mean)
    cl1_n = n_total
    tot_var = cl1_n*cl1_var
    thr1 = 0
    thr2 = 0
    for i in range(0, n_total):
        n_thr1 = ordr[i]
        n_cl1_mean, n_cl1_n = delItemMean(cl1_mean, cl1_n, n_thr1, 1)
        n_cl1_mean_sqr, n_cl1_n = delItemMean(cl1_mean_sqr, cl1_n, n_thr1**2,
                                              1)
        n_cl1_var = varFromMeans(n_cl1_mean_sqr, n_cl1_mean)
        sub_ordr = ordr[0:i+1]
        n_thr2, sub_var = biThresh(sub_ordr)
        n_tot_var = n_cl1_n*n_cl1_var + sub_var
        if n_tot_var > tot_var:
            tot_var = n_tot_var
            thr1 = n_thr1
            thr2 = n_thr2
        cl1_mean = n_cl1_mean
        cl1_mean_sqr = n_cl1_mean_sqr
        cl1_var = n_cl1_var
        cl1_n = n_cl1_n
    return thr1, thr2
