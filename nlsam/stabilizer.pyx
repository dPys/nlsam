#cython: wraparound=False, cdivision=True, boundscheck=False, language_level=3, embedsignature=True

cimport cython
from libc.math cimport sqrt, exp, fabs, M_PI
from scipy.special.cython_special cimport ndtri, gamma, chndtr


# libc.math isnan does not work on windows, it is called _isnan, so we use this one instead
# same thing for NAN apparently
cdef extern from "numpy/npy_math.h" nogil:
    bint npy_isnan(double x)
    double NPY_NAN

cdef extern from "hyp_1f1.h" nogil:
    double gsl_sf_hyperg_1F1(double a, double b, double x)

cdef double hyp1f1(double a, double b, double x) nogil:
    """Wrapper for 1F1 hypergeometric series function
    http://en.wikipedia.org/wiki/Confluent_hypergeometric_function"""
    return gsl_sf_hyperg_1F1(a, b, x)


cpdef double chi_to_gauss(double m, double eta, double sigma, double N, double alpha=0.0001, bint use_nan=False) nogil:
    """Maps the noisy signal intensity from a Rician/Non central chi distribution
    to its gaussian counterpart. See p. 4 of [1] eq. 12.

    Input
    --------
    m : double
        The noisy, Rician/Non central chi distributed value
    eta : double
        The underlying signal intensity estimated value
    sigma : double
        The gaussian noise estimated standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)
    alpha : double
        Confidence interval for the cumulative distribution function.
        Clips the cdf to alpha/2 <= cdf <= 1-alpha/2
    use_nan : bool
        If True, returns nans values when outside the confidence interval specified by alpha
        instead of clipping the outliers values to alpha.

    Return
    --------
        double : The noisy gaussian distributed signal intensity

    Reference
    -----------
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """
    cdef double cdf, inv_cdf_gauss

    cdf = 1. - _marcumq_cython(eta/sigma, m/sigma, N)

    # clip cdf between alpha/2 and 1-alpha/2
    if cdf < alpha/2:
        if use_nan:
            cdf = NPY_NAN
        else:
            cdf = alpha/2
    elif cdf > 1 - alpha/2:
        if use_nan:
            cdf = NPY_NAN
        else:
            cdf = 1 - alpha/2

    inv_cdf_gauss = eta + sigma * ndtri(cdf)
    return inv_cdf_gauss


cdef double _marcumq_cython(double a, double b, double M, double eps=1e-8) nogil:
    """Computes the generalized Marcum Q function of order M.
    http://en.wikipedia.org/wiki/Marcum_Q-function

    a : double, eta/sigma
    b : double, m/sigma
    M : double, order of the function (Number of coils, N=1 for Rician noise)

    return : double
        Value of the function, always between 0 and 1 since it's a pdf.

    Notes
    ------
    We actually use chndtr which is the cdf of a chi square variable with a few
    change of arguments. The relation is

    nchi2_pdf(x, k, lbda) = 1 - Marcum(sqrt(lbda), sqrt(x), k/2)

    and therefore

    Marcum(a, b, M) = 1 - nchi2_pdf(b**2, a**2, 2*M)

    or in our notation

    nchi_pdf = 1 - Marcum(a**2, b**2, 2*M)
    """
    cdef:
        double k = 2 * M
        double lbda = a**2
        double x = b**2
        double out

    if fabs(b) < eps:
        return 1.

    if fabs(M) < eps:
        k = eps

    # a, b and M can not be negative normally in our case,
    # but a is allowed and the real marcumq function is symmetric in a and b apparently
    # for general values of M. b and M negative have no physical sense though in our case.
    if M < 0:
        return 1.

    out = 1. - chndtr(x, k, lbda)
    return out


cpdef double fixed_point_finder(double m_hat, double sigma, double N,
    bint clip_eta=True, int max_iter=100, double eps=1e-6) nogil:
    """Fixed point formula for finding eta. Table 1 p. 11 of [1]

    Input
    --------
    m_hat : double
        Initial value for the estimation of eta
    sigma : double
        Gaussian standard deviation of the noise
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)
    max_iter : int, default=100
        Maximum number of iterations before breaking from the loop
    eps : double, default = 1e-6
        Criterion for reaching convergence between two subsequent estimates
    clip_eta : bool, default True
        If True, eta is clipped to 0 when below the noise floor (Bai 2014).
        If False, a new starting point m_hat is used and yields a negative eta value,
        which ensures symmetry of the normal distribution near 0 (Koay 2009).

        Having eta at zero is coherent with magnitude values being >= 0,
        but allowing negative eta is in line with the original framework
        and allows averaging of normally distributed values.

    Return
    -------
    t1 : double
        Estimation of the underlying signal value
    """
    cdef:
        double delta, m, t0, t1

    # If m_hat is below the noise floor, return 0 instead of negatives
    # as per Bai 2014
    if clip_eta and (m_hat < sqrt(0.5 * M_PI) * sigma):
        return 0.0

    delta = _beta(N) * sigma - m_hat

    if fabs(delta) < 1e-15:
        return 0.0

    if delta > 0:
        m = _beta(N) * sigma + delta
    else:
        m = m_hat

    t0 = m

    for _ in range(max_iter):

        t1 = _fixed_point_k(t0, m, sigma, N)

        if fabs(t1 - t0) < eps:
            break

        t0 = t1

    if npy_isnan(t1):  # Should not happen unless numerically unstable
        t1 = 0.0

    if (delta > 0) and (not clip_eta):
        return -t1
    else:
        return t1


cdef double _beta(double N) nogil:
    """Helper function for xi, see p. 3 [1] just after eq. 8.
    Generalized version for non integer N"""
    return sqrt(2) * gamma(N + 0.5) / gamma(N)


cdef double _fixed_point_k(double eta, double m, double sigma, double N) nogil:
    """Helper function for fixed_point_finder, see p. 11 [1] eq. D2."""
    cdef:
        double fpg, num, denom, h1f1m, h1f1p
        double eta2sigma = -eta**2/(2*sigma**2)

    fpg = sqrt(m**2 + (xi(eta, sigma, N) - 2*N) * sigma**2)
    h1f1m = hyp1f1(-0.5, N, eta2sigma)
    h1f1p = hyp1f1(0.5, N+1, eta2sigma)

    num = fpg * (fpg - eta)
    denom = eta * (1 - (_beta(N)**2 / (2*N)) * h1f1m * h1f1p) - fpg

    return eta - num / denom


cdef double _fixed_point_k_v2(double eta, double m, double sigma, double N) nogil:
    """Helper function for fixed_point_finder, see p. 11 [1] eq. D3.

    This is a secret equation scheme which gives rise to a different fixed point iteration
    and is only here for completion purposes, as it a replacement for eq. D2.
    Consider this as a secret bonus for looking at the code since we currently do not use it ;)"""
    cdef:
        double num, denom
        double eta2sigma = -eta**2/(2*sigma**2)
        double beta_N = _beta(N)
        double h1f1m = hyp1f1(-0.5, N, eta2sigma)
        double h1f1p = hyp1f1(0.5, N+1, eta2sigma)

    num = 2 * N * sigma * (m - beta_N * sigma * h1f1m)
    denom = beta_N * eta * h1f1p

    return eta + num / denom

cpdef double xi(double eta, double sigma, double N) nogil:
    """Standard deviation scaling factor formula, see p. 3 of [1], eq. 10.

    Input
    -------
    eta : double
        Signal intensity
    sigma : double
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)

    Return
    --------
    double
        The correction factor xi, where sigma_gaussian**2 = sigma**2 / xi
    """

    cdef:
        double h1f1, out
        double eta2sigma = eta**2/sigma**2

    if fabs(sigma) < 1e-15 or check_high_SNR(eta, sigma, N):
        return 1.

    h1f1 = hyp1f1(-0.5, N, -eta2sigma/2)
    out = 2*N + eta2sigma - (_beta(N) * h1f1)**2

    # Ridiculous SNR > 1e15 returns nonsense high numbers (positive or negative in the order of 1e30).
    # due to floating point precision issues, but the function is bounded between 0 and 1.
    # It starts to accumulate error around SNR ~ 1e4 though,
    # so we clip it to 1 (if needed) to stay on the safe side.

    if fabs(out) > 1:
        out = 1.0

    return out


# Helper function for the root finding loop
cdef double k(double theta, double N, double r) nogil:
    cdef:
        # Fake SNR value for xi
        double eta = theta
        double sigma = 1.
        double g, h1f1m, h1f1p, num, denom

    g = sqrt(xi(eta, sigma, N) * (1 + r**2) - 2*N)
    h1f1m = hyp1f1(-0.5, N, -theta**2/2)
    h1f1p = hyp1f1(0.5, N+1, -theta**2/2)

    num = g * (g - theta)
    denom = theta * (1 + r**2) * (1 - _beta(N)**2/(2*N) * h1f1m * h1f1p) - g

    return theta - num / denom


cpdef double root_finder(double r, double N, int max_iter=500, double eps=1e-6) nogil:
    cdef:
        double lower_bound = sqrt((2*N / xi(0, 1, N)) - 1)
        double t0, t1

    if r < lower_bound:
        return 0

    t0 = r - lower_bound
    t1 = k(t0, N, r)

    for _ in range(max_iter):

        t0 = t1
        t1 = k(t0, N, r)

        if fabs(t1 - t0) < eps:
            break

    return t1


cdef double check_high_SNR(double eta, double sigma, double N) nogil:
    '''If the SNR is high enough against N, these corrections factors change basically nothing, so may as well return early.'''
    cdef:
        double SNR = eta / sigma

    if N < 4.0:
        # xi = 0.991358489912443
        return SNR > 20.0
    elif (N > 4.0) and (N < 12.0):
        # xi = 0.9929057132635535
        return SNR > 40.0
    elif (N > 12.0) and (N < 48.0):
        # xi = 0.9952937754769664
        return SNR > 100.0
    else:
        return False


# Test for cython functions
def _test_marcumq_cython(a, b, M):
    return _marcumq_cython(a, b, M)


def _test_beta(N):
    return _beta(N)


def _test_fixed_point_k(eta, m, sigma, N):
    return _fixed_point_k(eta, m, sigma, N)
