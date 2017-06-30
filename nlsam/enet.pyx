#cython: wraparound=False, cdivision=True, boundscheck=False

import cython
import numpy as np
cimport numpy as np
from _glmnet import elnet, solns
# from numba import jit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


cdef double[:,:] elastic_net_path(double X, y, double rho, bint ols=False, int nlam=100, bint fit_intercept=False, bint pos=False, bint standardize=False) nogil:
    """return full path for ElasticNet"""

    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elastic_net(X, y, rho,
                                                            nlam=nlam,
                                                            fit_intercept=fit_intercept,
                                                            pos=pos,
                                                            standardize=standardize)
    nobs, nx = X.shape

    a0 = a0[:lmu]
    ca = ca[:nx, :lmu]
    ia = ia[:nx]
    nin = nin[:lmu]
    rsq = rsq[:lmu]
    alm = alm[:lmu]

    if len(nin) == 0:
        ninmax = 0
    else:
        ninmax = max(nin)

    if ninmax == 0:
        return np.zeros((nobs, nlam), dtype=np.float64), np.zeros([nx, nlam], dtype=np.float64)

    ca = ca[:ninmax]
    df = np.sum(ca != 0, axis=0)
    ja = ia[:ninmax] - 1    # ia is 1-indexed in fortran

    oja = np.argsort(ja)
    ja1 = ja[oja]
    beta = np.zeros([nx, lmu], dtype=np.float64)
    beta[ja1] = ca[oja]

    return beta


cdef select_best_path(double[:,:] X, double[:] y, double[:,:] beta, double[:,:] mu, variance=None, criterion='aic') nogil=True:
    '''See https://arxiv.org/pdf/0712.0881.pdf p. 9 eq. 2.15 and 2.16

    With regards to my notation :
    X is D, the regressor/dictionary matrix
    y is X, the measured signal we wish to reconstruct
    beta is alpha, the coefficients
    mu is the denoised reconstruction X_hat = D * alpha
    '''

    y = y.ravel()
    n = y.shape[0]
    p = X.shape[1]

    if criterion == 'aic':
        w = 2
    elif criterion == 'bic':
        w = np.log(n)
    elif criterion == 'ric':
        w = 2 * np.log(p)
    else:
        raise ValueError('Criterion {} is not supported!'.format(criterion))


    mse = np.mean((y[..., None] - mu)**2, axis=0)
    rss = np.sum((y[..., None] - mu)**2, axis=0)
    df_mu = np.sum(beta != 0, axis=0) #, dtype=np.int32)

    # Use mse = SSE/n estimate for sample variance - we assume normally distributed
    # residuals though for the log-likelihood function...
    if variance is None:
        criterion = n * np.log(mse) + df_mu * w
    else:
        criterion = rss / (n * variance) + (w * df_mu / n)

    # We don't want empty models
    criterion[df_mu == 0] = 1e300
    best_idx = np.argmin(criterion, axis=0)

    # We can only estimate sigma squared after selecting the best model
    if n > df_mu[best_idx]:
        estimated_variance = np.sum((y - mu[:, best_idx])**2) / (n - df_mu[best_idx])
    else:
        estimated_variance = 0

    return mu[:, best_idx], beta[:, best_idx], best_idx


def lasso_path(X, y, ols=False, nlam=100, fit_intercept=False, pos=False, standardize=False):
    """return full path for Lasso"""

    return elastic_net_path(X, y, rho=1.0, ols=ols, nlam=nlam, fit_intercept=fit_intercept, pos=pos, standardize=standardize)


cdef elastic_net(X, y, rho, bint pos=False, double thr=1e-4, weights=None, vp=None, bint copy=True,
                bint standardize=False, int nlam=100, int maxit=1e5, bintfit_intercept=False) nogil=True:
    """
    Raw-output wrapper for elastic net linear regression.
    X is D
    y is X
    rho for lasso/elastic net tradeoff
    """

    if copy:
        # X/y is overwritten in the fortran function at every loop, so we must copy it each time
        X = np.array(X, copy=True, dtype=np.float64, order='F')
        y = np.array(y, copy=True, dtype=np.float64, order='F').ravel()

    jd = np.zeros(1)        # X to exclude altogether from fitting
    ulam = None             # User-specified lambda values

    box_constraints = np.zeros((2, X.shape[1]), order='F')
    box_constraints[1] = 9.9e35 # this is a large number in fortran

    if not pos:
        box_constraints[0] = -9.9e35

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(X.shape[0], order='F')
    else:
        weights = np.array(weights, copy=True, order='F')

    # Uniform penalties if none were specified.
    if vp is None:
        vp = np.ones(X.shape[1], order='F')
    else:
        vp = vp.copy()

    # Call the Fortran wrapper.
    nx = X.shape[1] + 1
    # Trick from official wrapper
    if X.shape[0] < X.shape[1]:
        flmin = 0.01
    else:
        flmin = 1e-4

    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elnet(rho,
                                                      X,
                                                      y,
                                                      weights,
                                                      jd,
                                                      vp,
                                                      box_constraints,
                                                      nx,
                                                      flmin,
                                                      ulam,
                                                      thr,
                                                      nlam=nlam,
                                                      isd=standardize,
                                                      maxit=maxit,
                                                      intr=fit_intercept)

    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr
