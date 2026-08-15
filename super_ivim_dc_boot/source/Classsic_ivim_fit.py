# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:58:05 2022

@author: Yael Zaffrani
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#import nlopt
from warnings import warn


D_factor = 0.1
s0_factor = np.sqrt(1000)


# bounds = [[0.0003, 0.009, 0.001, 50],[0.01, 0.04,0.5, 300]] # d,d*,f,so

# D_factor = 20000#1 / np.sqrt(bounds[1][0] - bounds[0][0])
# f_factor = 80#1 / np.sqrt(bounds[1][1] - bounds[0][1])
# DStar_factor = 200#1 / np.sqrt(bounds[1][2] - bounds[0][2])
# s0_factor = 1#1 / np.sqrt(bounds[1][3] - bounds[0][3])


def IVIM_model(b_vector, D, DStar, f, s0):
    #si = s0 * (f * np.exp(-b_vector * (D + DStar)) + (1 - f) * np.exp(-b_vector * D))
    si = s0 * (f * np.exp(-b_vector * (D+ DStar)) + (1 - f) * np.exp(-b_vector * D))
    return si


def ivimN(b_vector, D, DStar, f, s0):
    # IVIM function with equal variance in the different IVIM parameters
    # D - fitted D (D_hat)
    # s0 - fitted s0 (s0_hat)
    D = D / D_factor
    s0 = s0 / s0_factor
    #noam
    # DStar = DStar / DStar_factor
    # f = f / f_factor
    
    return s0 * (f * np.exp(-b_vector * (D + DStar)) + (1 - f) * np.exp(-b_vector * D))


def parmNRMSE(org_param, fit_param, del_index, dim=0):
    org_param = np.delete(org_param, del_index, dim)

    return (np.linalg.norm(org_param - fit_param) / np.sqrt(len(org_param)))/np.mean(org_param)


def ivim_fit_nlls_error(xData, gradData, b_vector, si):
    D, DStar, f, s0_h = xData
    si_fit = ivimN(b_vector, D, DStar, f, s0_h)
    return np.mean((si - si_fit) ** 2)


def fit_least_squares_lm(N, b_vector, si, p0):
    # the algorithm uses the Levenberg-Marquardt algorithm
    p0[0] *= D_factor
    p0[3] *=s0_factor
    del_index = []
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])

    for i in range(N):
        s = np.squeeze(si[:, i])
        try:
            params = curve_fit(ivimN, b_vector, s, p0, maxfev=2000)
            D = np.append(D, params[0][0] / D_factor)
            DStar = np.append(DStar, params[0][1])
            f = np.append(f, params[0][2])
            s0 = np.append(s0, params[0][3] / s0_factor)
        except:
            # warn(f"{i} lm fit failed. removing sample from DS")
            del_index.append(i)

    return D, DStar, f, s0, del_index


def fit_least_squares_trf(N, b_vector, si, bounds, p0):
    # the algorithm uses the Trust Region Reflective algorithm
    p0[0] *= D_factor
    p0[3] *= s0_factor
    bounds_factor = ([bounds[0][0] * D_factor, bounds[0][1], bounds[0][2], bounds[0][3] * s0_factor],
                     [bounds[1][0] * D_factor, bounds[1][1], bounds[1][2], bounds[1][3] * s0_factor])
    del_index = []
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])

    for i in range(N):
        s = np.squeeze(si[:, i])
        try:
            params = curve_fit(ivimN, b_vector, s, p0, bounds=bounds_factor, maxfev=30000)
            D = np.append(D, params[0][0] / D_factor)
            DStar = np.append(DStar, params[0][1])
            f = np.append(f, params[0][2])
            s0 = np.append(s0, params[0][3] / s0_factor)
        except:
            # warn(f"{i} trf fit failed. removing sample from DS")
            del_index.append(i)
    return D, DStar, f, s0, del_index


def fit_least_squers_BOBYQA(N, b_vector, si, bounds, p0):
    p0[0] *= D_factor
    p0[3] *= s0_factor
    D, DStar, f, s0 = p0
    bounds = ([bounds[0][0] * D_factor, bounds[0][1], bounds[0][2], bounds[0][3] * s0_factor],
              [bounds[1][0] * D_factor, bounds[1][1], bounds[1][2], bounds[1][3] * s0_factor])
    lb = np.asarray(bounds[0])
    ub = np.asarray(bounds[1])
    opt = nlopt.opt(nlopt.LN_BOBYQA, 4)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    #
    opt.set_maxeval(2000)
    opt.set_ftol_abs(0.00001)

    del_index = []
    if f > 0.0 and D > 0.0:
        D = np.array([])
        DStar = np.array([])
        f = np.array([])
        s0 = np.array([])
        for i in range(N):
            s = np.squeeze(si[:, i])
            try:
                opt.set_min_objective(
                    lambda x, grad: ivim_fit_nlls_error(xData=x, gradData=grad, b_vector=b_vector, si=s))
                xopt = opt.optimize(p0)
                D = np.append(D, xopt[0] / D_factor)
                DStar = np.append(DStar, xopt[1])
                f = np.append(f, xopt[2])
                s0 = np.append(s0, xopt[3] / s0_factor)
            except:
                # warn(f"{i} bobyqa fit failed. removing sample from DS")
                del_index.append(i)
    return D, DStar, f, s0, del_index


def IVIM_fit_sls(N, si, b_vector, bounds, p0, min_bval_high=200):
    # for one smaple
    # first estimate D,S0 for mono-exp:
    p0[0] *= D_factor
    p0[3] *= s0_factor
    # p0[1] *= DStar_factor
    # p0[2] *= f_factor
    
    s_high = si[b_vector >= min_bval_high, :]
    b_vector_high = b_vector[b_vector >= min_bval_high]
    s0_d, D = fitMonoExpModel(s_high, b_vector_high)

    s0 = si[0, :]

    f = (s0 - s0_d) / s0
    # find D* by NLLS of IVIM:
    # si_remaining = si - s0*(1 - f) * np.exp(-b_vector_high * Dt)
    bounds_Ds = (bounds[0][1], bounds[1][1])
    p0_Ds = p0[1]
    del_index = []
    # params, _ = curve_fit(lambda b, Dp: Fp * np.exp(-b * Dp), b_vector, si_remaining, p0=p0_Ds, bounds=bounds_Ds)
    DStar = np.array([])
    for i in range(N):
        s = np.squeeze(si[:, i])
        try:
            params, _ = curve_fit(lambda b, DStar: s0[i] * (
                    f[i] * np.exp(-b_vector * (D[i] + DStar)) + (1 - f[i]) * np.exp(-b_vector * D[i])), b_vector, s,
                                  p0=p0_Ds, bounds=bounds_Ds, maxfev=1000)
            DStar = np.append(DStar, params[0])
        except:
            # warn(f"{i} sls fit failed. removing sample from DS")
            del_index.append(i)

    return D, DStar, f, s0, s0_d, del_index

def fitMonoExpModel(s, b_vector):
    A = np.matrix(
        np.concatenate((np.ones((len(b_vector), 1), order='C'), -np.reshape(b_vector, (len(b_vector), 1))), axis=1))
    s = np.log(s)
    s[np.isinf(s)] = 0.0
    x = np.linalg.lstsq(A, s, rcond=None)

    s0 = np.exp(x[0][0])
    ADC = x[0][1]

    return s0, ADC


def IVIM_fit_sls_lm(N, si, b_vector, bounds, p0, min_bval_high=200):
    D_sls, DStar_sls, f_sls, s0_sls, s0_d, del_index = IVIM_fit_sls(N, si, b_vector, bounds, p0, min_bval_high)
    si = np.delete(si, del_index, 1)
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])
    del_index = []
    for i in range(N):
        p0 = [D_sls[i] , DStar_sls[i], f_sls[i], s0_sls[i] ]
        try:
            D_fit, DStar_fit, f_fit, s0_fit, fit_del_index = fit_least_squares_lm(1, b_vector, si[:, i, None], p0)
            if len(fit_del_index) != 0:
                del_index.append(i)
            D = np.append(D, D_fit)
            DStar = np.append(DStar, DStar_fit)
            f = np.append(f, f_fit)
            s0 = np.append(s0, s0_fit)
        except:
            # warn(f"{i} sls-lm fit failed. removing sample from DS")
            del_index.append(i)
    return D, DStar, f, s0, s0_d, del_index

def IVIM_fit_sls_trf(N, si, b_vector, bounds, p0, eps=0.00001, min_bval_high=200):
    D_sls, DStar_sls, f_sls, s0_sls, s0_d, del_index = IVIM_fit_sls(N, si, b_vector, bounds, p0, min_bval_high)
    print(DStar_sls)
    si = np.delete(si, del_index, 1)
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])
    del_index = []
    for i in range(N):
        p0 = [D_sls[i] , DStar_sls[i], f_sls[i], s0_sls[i] ]
        # if p0 out of bound, take extrime bound as p0:
        sls_bounds = np.array(bounds)  # d,d*,f,so
        for j in range(len(p0)):
            if p0[j] < sls_bounds[0, j]:
                p0[j] = sls_bounds[0, j] + sls_bounds[0, j] * eps
            elif p0[j] > sls_bounds[1, j]:
                p0[j] = sls_bounds[1, j] - sls_bounds[0, j] * eps
        try:
            D_fit, DStar_fit, f_fit, s0_fit, fit_del_index = fit_least_squares_trf(1, b_vector, si[:, i, None], bounds,
                                                                                   p0)
            if len(fit_del_index) != 0:
                del_index.append(i)
            D = np.append(D, D_fit)
            DStar = np.append(DStar, DStar_fit)
            f = np.append(f, f_fit)
            s0 = np.append(s0, s0_fit)
        except:
            # warn(f"{i} sls-lm fit failed. removing sample from DS")
            del_index.append(i)
    return D, DStar, f, s0, s0_d, del_index


def IVIM_fit_sls_BOBYQA(N, si, b_vector, bounds, p0, eps=0.00001, min_bval_high=200):
    D_sls, DStar_sls, f_sls, s0_sls, s0_d, del_index = IVIM_fit_sls(N, si, b_vector, bounds, p0, min_bval_high)
    si = np.delete(si, del_index, 1)
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])
    del_index = []
    for i in range(N):
        p0 = [D_sls[i] , DStar_sls[i], f_sls[i], s0_sls[i] ]
        # if p0 out of bound, take extrime bound as p0:
        sls_bounds = np.array(bounds)  # d,d*,f,so
        for j in range(len(p0)):
            if p0[j] < sls_bounds[0, j]:
                p0[j] = sls_bounds[0, j] + sls_bounds[0, j] * eps
            elif p0[j] > sls_bounds[1, j]:
                p0[j] = sls_bounds[1, j] - sls_bounds[0, j] * eps
        try:
            D_fit, DStar_fit, f_fit, s0_fit, fit_del_index = fit_least_squers_BOBYQA(1, b_vector, si[:, i, None], bounds,
                                                                                   p0)
            if len(fit_del_index) != 0:
                del_index.append(i)
            D = np.append(D, D_fit)
            DStar = np.append(DStar, DStar_fit)
            f = np.append(f, f_fit)
            s0 = np.append(s0, s0_fit)
        except:
            # warn(f"{i} sls-lm fit failed. removing sample from DS")
            del_index.append(i)
    return D, DStar, f, s0, s0_d, del_index

def plot_signal_fit(b_vector, si, si_fit, si_original, s0, D, DStar, f):
    '''
    plot log(si/s0) with respect to b-value
    :param signal:
    :param b_vector:
    :return:
    '''
    log_signal_org = np.log(si_original / si_original[0])
    log_signal = np.log(si / si[0])
    log_signal_fit = np.log(si_fit / si_fit[0])
    fig, ax = plt.subplots()

    ax.plot(b_vector, log_signal_org, c='g', marker='*')
    ax.plot(b_vector, log_signal, c='r', marker='*')
    ax.plot(b_vector, log_signal_fit, c='b', marker='*')

    ax.set_title(f'IVIM Signal: D = {D:.3f}, D* = {DStar:.2f}, f = {f:.2f}, s0 = {s0:.2f}')

    ax.set_xlabel('b-value')
    ax.set_ylabel('log(s_i/s_0)')
    ax.legend(['GT', f'Si_noisy', 'fit'])

    plt.show()
