import io
import copy
import unittest
import warnings
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bs4 import BeautifulSoup
from pandas.testing import assert_frame_equal
from tobit.tobit_reg import Tobit

warnings.filterwarnings("ignore")


tbl0 = pd.DataFrame(
    {
        "y": {
            "Model:": "Tobit",
            "Method:": "Least Squares",
            "Date:": "Tue, 27 Aug 2024",
            "Time:": "02:31:21",
            "No. Observations:": "10000",
            "Df Residuals:": "9997",
            "Df Model:": "2",
            "Covariance Type:": "HC1",
        },
        "R-squared:": {
            "Model:": "Adj. R-squared:",
            "Method:": "F-statistic:",
            "Date:": "Prob (F-statistic):",
            "Time:": "Log-Likelihood:",
            "No. Observations:": "AIC:",
            "Df Residuals:": "BIC:",
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
        "0.794": {
            "Model:": 0.794,
            "Method:": 16900.0,
            "Date:": 0.0,
            "Time:": -20373.0,
            "No. Observations:": 40750.0,
            "Df Residuals:": 40770.0,
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
    }
)


tbl1 = pd.DataFrame(
    {
        "coef": {0: 7.2177, 1: 0.236, 2: 1.1856},
        "std err": {0: 0.074, 1: 0.006, 2: 0.007},
        "z": {0: 97.731, 1: 37.688, 2: 180.458},
        "P>|z|": {0: 0.0, 1: 0.0, 2: 0.0},
        "[0.025": {0: 7.073, 1: 0.224, 2: 1.173},
        "0.975]": {0: 7.362, 1: 0.248, 2: 1.198},
    }
)

tbl2 = pd.DataFrame(
    {
        "1.221": {"Prob(Omnibus):": 0.543, "Skew:": -0.027, "Kurtosis:": 2.987},
        "Durbin-Watson:": {
            "Prob(Omnibus):": "Jarque-Bera (JB):",
            "Skew:": "Prob(JB):",
            "Kurtosis:": "Cond. No.",
        },
        "1.998": {"Prob(Omnibus):": 1.237, "Skew:": 0.539, "Kurtosis:": 45.2},
    }
)
tbl2.index.name = "Omnibus:"

tbl3 = pd.DataFrame(
    {
        "y": {
            "Model:": "Tobit",
            "Method:": "Least Squares",
            "Date:": "Fri, 28 Nov 2025",
            "Time:": "09:55:22",
            "No. Observations:": "10000",
            "Df Residuals:": "9997",
            "Df Model:": "2",
            "Covariance Type:": "HC1",
        },
        "Pseudo R-squared:": {
            "Model:": float("nan"),
            "Method:": "F-statistic:",
            "Date:": "Prob (F-statistic):",
            "Time:": "Log-Likelihood:",
            "No. Observations:": "AIC:",
            "Df Residuals:": "BIC:",
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
        "0.784": {
            "Model:": float("nan"),
            "Method:": 14600.0,
            "Date:": 0.0,
            "Time:": -19385.0,
            "No. Observations:": 38780.0,
            "Df Residuals:": 38800.0,
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
    }
)

tbl4 = pd.DataFrame(
    {
        "coef": {0: 6.2387, 1: 0.258, 2: 1.2994},
        "std err": {0: 0.08, 1: 0.006, 2: 0.008},
        "z": {0: 77.934, 1: 40.197, 2: 168.484},
        "P>|z|": {0: 0.0, 1: 0.0, 2: 0.0},
        "[0.025": {0: 6.082, 1: 0.245, 2: 1.284},
        "0.975]": {0: 6.396, 1: 0.271, 2: 1.315},
    }
)

tbl5 = pd.DataFrame(
    {
        "26.544": {"Prob(Omnibus):": 0.0, "Skew:": -0.036, "Kurtosis:": 3.272},
        "Durbin-Watson:": {
            "Prob(Omnibus):": "Jarque-Bera (JB):",
            "Skew:": "Prob(JB):",
            "Kurtosis:": "Cond. No.",
        },
        "1.975": {"Prob(Omnibus):": 32.908, "Skew:": 7.15e-08, "Kurtosis:": 45.2},
    }
)

tbl6 = pd.DataFrame(
    {
        "y": {
            "Model:": "Tobit",
            "Method:": "Least Squares",
            "Date:": "Fri, 28 Nov 2025",
            "Time:": "09:57:42",
            "No. Observations:": "10000",
            "Df Residuals:": "9997",
            "Df Model:": "2",
            "Covariance Type:": "HC1",
        },
        "Pseudo R-squared:": {
            "Model:": float("nan"),
            "Method:": "F-statistic:",
            "Date:": "Prob (F-statistic):",
            "Time:": "Log-Likelihood:",
            "No. Observations:": "AIC:",
            "Df Residuals:": "BIC:",
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
        "0.783": {
            "Model:": float("nan"),
            "Method:": 14280.0,
            "Date:": 0.0,
            "Time:": -19301.0,
            "No. Observations:": 38610.0,
            "Df Residuals:": 38630.0,
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
    }
)

tbl7 = pd.DataFrame(
    {
        "coef": {0: 6.5611, 1: 0.2597, 2: 1.3073},
        "std err": {0: 0.08, 1: 0.006, 2: 0.008},
        "z": {0: 81.997, 1: 40.367, 2: 166.822},
        "P>|z|": {0: 0.0, 1: 0.0, 2: 0.0},
        "[0.025": {0: 6.404, 1: 0.247, 2: 1.292},
        "0.975]": {0: 6.718, 1: 0.272, 2: 1.323},
    }
)

tbl8 = pd.DataFrame(
    {
        "30.220": {"Prob(Omnibus):": 0.0, "Skew:": -0.037, "Kurtosis:": 3.294},
        "Durbin-Watson:": {
            "Prob(Omnibus):": "Jarque-Bera (JB):",
            "Skew:": "Prob(JB):",
            "Kurtosis:": "Cond. No.",
        },
        "1.973": {"Prob(Omnibus):": 38.22, "Skew:": 5.02e-09, "Kurtosis:": 45.2},
    }
)

tbl9 = pd.DataFrame(
    {
        "y": {
            "Model:": "Tobit",
            "Method:": "Least Squares",
            "Date:": "Fri, 28 Nov 2025",
            "Time:": "09:48:28",
            "No. Observations:": "10000",
            "Df Residuals:": "9997",
            "Df Model:": "2",
            "Covariance Type:": "HC1",
        },
        "Pseudo R-squared:": {
            "Model:": float("nan"),
            "Method:": "F-statistic:",
            "Date:": "Prob (F-statistic):",
            "Time:": "Log-Likelihood:",
            "No. Observations:": "AIC:",
            "Df Residuals:": "BIC:",
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
        "0.746": {
            "Model:": float("nan"),
            "Method:": 11130.0,
            "Date:": 0.0,
            "Time:": -17985.0,
            "No. Observations:": 35980.0,
            "Df Residuals:": 36000.0,
            "Df Model:": float("nan"),
            "Covariance Type:": float("nan"),
        },
    }
)

tbl10 = pd.DataFrame(
    {
        "coef": {0: 5.209, 1: 0.292, 2: 1.4749},
        "std err": {0: 0.094, 1: 0.007, 2: 0.01},
        "z": {0: 55.498, 1: 41.846, 2: 147.829},
        "P>|z|": {0: 0.0, 1: 0.0, 2: 0.0},
        "[0.025": {0: 5.025, 1: 0.278, 2: 1.455},
        "0.975]": {0: 5.393, 1: 0.306, 2: 1.494},
    }
)

tbl11 = pd.DataFrame(
    {
        "145.139": {"Prob(Omnibus):": 0.0, "Skew:": -0.052, "Kurtosis:": 3.796},
        "Durbin-Watson:": {
            "Prob(Omnibus):": "Jarque-Bera (JB):",
            "Skew:": "Prob(JB):",
            "Kurtosis:": "Cond. No.",
        },
        "1.983": {"Prob(Omnibus):": 268.725, "Skew:": 4.44e-59, "Kurtosis:": 45.2},
    }
)

tbl0.index.name = "Dep. Variable:"
tbl3.index.name = "Dep. Variable:"
tbl6.index.name = "Dep. Variable:"
tbl9.index.name = "Dep. Variable:"

tbl2.index.name = "Omnibus:"
tbl5.index.name = "Omnibus:"
tbl8.index.name = "Omnibus:"
tbl11.index.name = "Omnibus:"


class TestTobit(unittest.TestCase):

    def test_tobit_regression(self):
        np.random.seed(123)
        # Generate the data

        # Number of observations
        n = 10000
        # Independent variables
        x1 = np.random.normal(loc=10, scale=3, size=n)
        x2 = np.random.normal(loc=5, scale=3, size=n)
        # Error term
        epsilon = np.random.normal(loc=0, scale=2, size=n)

        # Linear model before censoring
        y_star = 5 + 0.3 * x1 + 1.5 * x2 + epsilon

        y = copy.deepcopy(y_star)
        X = np.column_stack((x1, x2))  # noqa
        X = sm.add_constant(X)  # noqa

        y = pd.Series(y)
        X = pd.DataFrame(X)  # noqa

        c_lw_values = [np.quantile(y_star, 0.10), None]
        c_up_values = [np.quantile(y_star, 0.90), None]
        bool_values = [True, False]  # for reparam and ols_option
        combo = list(
            itertools.product(c_lw_values, c_up_values, bool_values, bool_values)
        )
        for y_l, y_u, reprm, ols_opt in combo:
            # Censoring
            if y_l is not None:
                y[y <= y_l] = y_l
            if y_u is not None:
                y[y >= y_u] = y_u

            # Separating instance from models allows access to class
            # attributes after model run.
            tobit = Tobit(
                y, X, reparam=reprm, c_lw=y_l, c_up=y_u, ols_option=ols_opt
            )  # noqa
            model = tobit.fit(cov_type="HC1", verbose=False)

            summary_html = model.summary().as_html()
            soup = BeautifulSoup(summary_html, "html.parser")
            tables = soup.find_all("table")
            results = [
                pd.read_html(io.StringIO(str(table)), header=0, index_col=0)[0]
                for table in tables
            ]

            if (y_l is None and y_u is None) or ols_opt:
                expected_results = [tbl0, tbl1, tbl2]
            elif y_l is not None and y_u is None:
                expected_results = [tbl3, tbl4, tbl5]
            elif y_l is None and y_u is not None:
                expected_results = [tbl6, tbl7, tbl8]
            else:
                expected_results = [tbl9, tbl10, tbl11]

            assert_frame_equal(
                results[0].drop("y", axis=1), expected_results[0].drop("y", axis=1)
            )
            [
                assert_frame_equal(r, s)
                for r, s in zip(results[1:], expected_results[1:])
            ]


if __name__ == "__main__":
    unittest.main()
