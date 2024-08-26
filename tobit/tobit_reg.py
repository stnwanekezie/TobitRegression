"""
Author: Stanley Nwanekezie
Created on: August 25, 2024
Last Modified: August 25, 2024
Email: stanwanekezie@gmail.com

Description:
    This is an implementation of the Tobit Regression model in Python with the
    flexibility for both upward and downward censoring of the dependent variable.
    The standard and reparameterized LLH function at https://en.wikipedia.org/wiki/Tobit_model
    have been adopted and extendedto include right censoring. Note that the reparameterized
    LLH is more robust according Olsen, Randall J. (1978). "Note on the Uniqueness of the Maximum
    Likelihood Estimator for the Tobit Model". Econometrica. 46 (5): 1211–1215. doi:10.2307/1911445.
    JSTOR 1911445. This implementation leverages the statsmodels module for robust data checking and
    result presentation capabilities. This ensures that model estimation results are presented
    in a familiar format. Parameter estimates using the standard LLH are validated against
    https://github.com/jamesdj/tobit/blob/master/tobit.py.
"""

import copy
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm
from statsmodels.api import OLS
from statsmodels.regression.linear_model import (
    OLSResults, # noqa
    RegressionResultsWrapper,
)
from statsmodels.tools.numdiff import approx_hess2


class Tobit(OLS):
    def __init__(
        self,
        endog,
        exog=None,
        reparam=True,
        c_lw=0,
        c_up=None,
        missing="none",
        hasconst=None,
        **kwargs
    ):
        self._c_lw = c_lw
        self._c_up = c_up
        self.reparam = reparam
        self.llh = 0
        self.scale = None
        self.params = None
        self.ols_params = None

        super().__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)
        if c_lw is not None and c_up is None:
            self.left_endog = endog[endog <= c_lw]
            self.left_exog = exog[np.where(endog <= c_lw, True, False), :]
            self.free_endog = endog[endog > c_lw]
            self.free_exog = exog[np.where(endog > c_lw, True, False), :]
        elif c_lw is None and c_up is not None:
            self.free_endog = endog[endog < c_up]
            self.free_exog = exog[np.where(endog < c_up, True, False), :]
            self.right_endog = endog[endog >= c_up]
            self.right_exog = exog[np.where(endog >= c_up, True, False), :]
        elif c_lw is not None and c_up is not None:
            self.left_endog = endog[endog <= c_lw]
            self.left_exog = exog[np.where(endog <= c_lw, True, False), :]
            self.free_endog = endog[(endog > c_lw) & (endog < c_up)]
            self.free_exog = exog[
                np.where((endog > c_lw) & (endog < c_up), True, False), :
            ]
            self.right_endog = endog[endog >= c_up]
            self.right_exog = exog[np.where(endog >= c_up, True, False), :]
        else:
            warnings.warn(
                "No censoring threshold provided; OLS will be used for model estimation."
            )
            self.__class__.__name__ = "OLS"

    @property
    def c_lw(self):
        return self._c_lw

    @property
    def c_up(self):
        return self._c_up

    @c_lw.setter
    def c_lw(self, new_value):
        raise AttributeError("Cannot modify lower threshold value after it is set.")

    @c_up.setter
    def c_up(self, new_value):
        raise AttributeError("Cannot modify upper threshold value after it is set.")

    def neg_llh_jac(self, params):
        scale, betas = params[0], params[1:]

        scale_jac_censored = 0
        betas_jac_censored = np.zeros(len(betas))
        if self._c_lw is not None:
            left_zscores = (self.left_endog - np.dot(self.left_exog, betas)) / scale
            left_d_dscale = np.dot(
                norm.pdf(left_zscores) / norm.cdf(left_zscores), -left_zscores / scale
            )
            left_d_dbetas = np.dot(
                norm.pdf(left_zscores) / norm.cdf(left_zscores), -self.left_exog / scale
            )
            scale_jac_censored += left_d_dscale
            betas_jac_censored += left_d_dbetas

        if self._c_up is not None:
            right_zscores = (np.dot(self.right_exog, betas) - self.right_endog) / scale
            right_d_dscale = np.dot(
                norm.pdf(right_zscores) / norm.cdf(right_zscores),
                -right_zscores / scale,
            )
            right_d_dbetas = np.dot(
                norm.pdf(right_zscores) / norm.cdf(right_zscores),
                self.right_exog / scale,
            )
            scale_jac_censored += right_d_dscale
            betas_jac_censored += right_d_dbetas

        free_zscores = (self.free_endog - np.dot(self.free_exog, betas)) / scale
        free_d_dscale = np.sum(free_zscores**2 - 1) / scale
        free_d_dbetas = np.dot(free_zscores, self.free_exog / scale)

        scale_jac = scale_jac_censored + free_d_dscale
        betas_jac = betas_jac_censored + free_d_dbetas

        return -np.append(scale_jac, betas_jac)

    def loglike(self, params):  # noqa
        if hasattr(self, "llh"):
            return self.llh
        else:
            return super().loglike(params)

    def neg_llh_func(self, params):
        scale, betas = params[0], params[1:]

        llf_censored = 0

        if self._c_lw is not None:
            llf_left = np.sum(
                log_ndtr((self.left_endog - np.dot(self.left_exog, betas)) / scale)
            )
            llf_censored += llf_left
        if self._c_up is not None:
            llf_right = np.sum(
                log_ndtr((np.dot(self.right_exog, betas) - self.right_endog) / scale)
            )
            llf_censored += llf_right

        llf_free = np.sum(
            norm.logpdf((self.free_endog - np.dot(self.free_exog, betas)) / scale)
            - np.log(max(scale, np.finfo("float").resolution))
        )

        self.llh = -1 * (llf_censored + llf_free)
        return self.llh

    # Reparameterization
    def neg_llh_func2(self, params):
        gamma, delta = params[0], params[1:]

        llf_censored = 0
        if self._c_lw is not None:
            llf_left = np.sum(
                norm.logcdf(gamma * self.left_endog - np.dot(self.left_exog, delta))
            )
            llf_censored += llf_left
        if self._c_up is not None:
            llf_right = np.sum(
                norm.logcdf(np.dot(self.right_exog, delta) - gamma * self.right_endog)
            )
            llf_censored += llf_right

        llf_free = np.sum(
            np.log(max(gamma, np.finfo("float").resolution))
            + norm.logpdf(gamma * self.free_endog - np.dot(self.free_exog, delta))
        )

        self.llh = -1 * (llf_censored + llf_free)
        return self.llh

    def neg_llh_jac2(self, params):
        gamma, delta = params[0], params[1:]

        gamma_jac_censored = 0
        delta_jac_censored = np.zeros(len(delta))

        if self._c_lw is not None:
            left_zscore = gamma * self.left_endog - np.dot(self.left_exog, delta)
            left_d_dgamma = np.dot(
                norm.pdf(left_zscore) / norm.cdf(left_zscore), self.left_endog
            )
            left_d_ddelta = np.dot(
                norm.pdf(left_zscore) / norm.cdf(left_zscore), -self.left_exog
            )
            delta_jac_censored += left_d_ddelta
            gamma_jac_censored += left_d_dgamma

        if self._c_up is not None:
            right_zscore = np.dot(self.right_exog, delta) - gamma * self.right_endog
            right_d_dgamma = np.dot(
                norm.pdf(right_zscore) / norm.cdf(right_zscore), -self.right_endog
            )
            right_d_ddelta = np.dot(
                norm.pdf(right_zscore) / norm.cdf(right_zscore), self.right_exog
            )
            delta_jac_censored += right_d_ddelta
            gamma_jac_censored += right_d_dgamma

        free_zscore = gamma * self.free_endog - np.dot(self.free_exog, delta)
        free_d_dgamma = np.sum(1 / gamma - free_zscore * self.free_endog)
        free_d_ddelta = np.dot(free_zscore, self.free_exog)

        gamma_jac = gamma_jac_censored + free_d_dgamma
        delta_jac = delta_jac_censored + free_d_ddelta

        return -np.append(gamma_jac, delta_jac)

    def fit_tobit(self, cov_type="HC1", cov_kwds=None, use_t=None, verbose=True):
        modl = super().fit(cov_type=cov_type)  # noqa
        self.ols_params = copy.deepcopy(modl.params)
        scale = np.sqrt(np.cov(modl.resid))  # noqa

        if self.reparam:
            gamma0 = 1 / np.sqrt(scale)
            delta0 = self.ols_params * gamma0
            params0 = np.append(gamma0, delta0)
            func = self.neg_llh_func2
            jac = self.neg_llh_jac2
        else:
            params0 = np.append(scale, self.ols_params)
            func = self.neg_llh_func
            jac = self.neg_llh_jac

        result = minimize(
            lambda params: func(params),
            params0,
            method="BFGS",
            jac=lambda params: jac(params),
            options={"disp": verbose},
        )

        if verbose:
            print(result)

        if self.reparam:
            self.scale = 1 / result.x[0] ** 2
            self.params = result.x[1:] / np.sqrt(self.scale)
        else:
            self.scale = result.x[0]
            self.params = result.x[1:]

        normalized_cov_params = np.linalg.inv(
            approx_hess2(np.append(self.scale, self.params), func)
        )
        self.normalized_cov_params = normalized_cov_params[  # noqa
            1:, 1:
        ]  # Removal of scale covariances

        lfit = OLSResults(
            self,
            self.params,
            normalized_cov_params=self.normalized_cov_params,
            scale=self.scale,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
        )

        return RegressionResultsWrapper(lfit)

    def fit(self, cov_type="HC1", cov_kwds=None, use_t=None, verbose=True, **kwargs):
        if all((self._c_lw is None, self._c_up is None)):
            return super().fit(cov_type=cov_type, cov_kwds=None, use_t=None, **kwargs)
        else:
            return self.fit_tobit(
                cov_type=cov_type, cov_kwds=None, use_t=None, verbose=True
            )


if __name__ == "__main__":
    rows, cols = 200, 5
    np.random.seed(42)
    y = np.random.randn(rows)
    X = sm.add_constant(np.random.randn(rows, cols))
    model = Tobit(y, X, reparam=False, c_lw=0, c_up=1).fit(cov_type="HC1")
    print(model.summary())
    model2 = OLS(y, X).fit(cov_type="HC1")
    print(model2.summary())
