"""Model functions for population fit
"""

import numpy as np
import math
from scipy.optimize import fsolve
from scipy.stats import gamma
import scipy.stats as st
import sys
from iminuit import cost,Minuit

t_factor_table = {}
t_factor_table[1] = 12.706
t_factor_table[2] = 4.303
t_factor_table[3] = 3.182
t_factor_table[3] = 2.776

def get_CI(T2,sT2):
    theta = (sT2**2)/T2
    k = T2/theta

    low = gamma.ppf(0.025,k,0,theta)
    up = gamma.ppf(0.975,k,0,theta)

    return (up - T2), (T2 - low)

def get_CI_t(T2,sT2,n):
    interval = sT2/math.sqrt(n)

    up = t_factor_table[n]*interval
    low = up

    return up, low

def gompertz(t,k,a,b):
    """Computes the population at given times using the Gompertz law, :math:`N = k e^{-e^{a-bt}}`.

    Parameters
    ----------
    t: array_like
        An array of times at which the population according to Gompertz law needs to be determined.

    k: float
        Maximum population value.

    a: float
        Gompertz law parameter determining the location of the inflection point during population growth.

    b: float
        Gompertz law population growth rate constant.

    Returns
    -------
    N: array_like
        Population at times t.
    """

    N = k*np.exp(-np.exp(a-b*t))

    return N

def compute_g_properties(k,a,b):
    # Inflection point
    x_inf = a/b
    y_inf = k/math.exp(1)

    print(x_inf)
    sys.stdout.flush()

    # Max growth
    gr_max = b*k/math.exp(1)

    return str(x_inf)+','+str(y_inf)+','+str(gr_max)

def gompertz_rates(t,k,a,b):
    """Computes the population growth rate at given times using the Gompertz law, :math:`N = k e^{-e^{a-bt}}`.

    Parameters
    ----------
    t: array_like
        An array of times at which the population according to Gompertz law needs to be determined.

    k: float
        Maximum population value.

    a: float
        Gompertz law parameter determining the location of the inflection point during population growth.

    b: float
        Gompertz law population growth rate constant.

    Returns
    -------
    rate: array_like
        Growth rate of the population at times t.
    """

    p = k*np.exp(-np.exp(a-b*t))

    rate = p*(b*np.exp(a-b*t))

    return rate

def d_simple_gompertz_rates(t,k,a,b):
    """Computes the derivative of the population growth rate at given times using the Gompertz law, :math:`N = k e^{-e^{a-bt}}`.

    Parameters
    ----------
    t: array_like
        An array of times at which the population according to Gompertz law needs to be determined.

    k: float
        Maximum population value.

    a: float
        Gompertz law parameter determining the location of the inflection point during population growth.

    b: float
        Gompertz law population growth rate constant.

    Returns
    -------
    rate_rate: array_like
        Time derivative of the population growth rate at times t.
    """
    p = k*np.exp(-np.exp(a-b*t))

    rate_1 = -(b**2)*p*np.exp(a-b*t)

    rate_2 = (b**2)*p*np.exp(a-b*t)*np.exp(a-b*t)

    rate_rate = rate_1 + rate_2

    return rate_rate

def compute_gompertz_inflections(t,k,a,b):
    """Computes the inflections points in the population growth using the Gompertz law, :math:`N = k e^{-e^{a-bt}}`.

    Parameters
    ----------
    t: array_like
        An array of times at which the population according to Gompertz law needs to be determined.

    k: float
        Maximum population value.

    a: float
        Gompertz law parameter determining the location of the inflection point during population growth.

    b: float
        Gompertz law population growth rate constant.

    Returns
    -------
    t1: float
        First inflection point, or the time at which the derivative of the population growth is at maximum. This occurs earlier than t2.

    t2: float
        Second inflection point, or the time at which the population growth is at maximum.
    """

    rate_rate = d_simple_gompertz_rates(t,k,a,b)
    rr_l = list(rate_rate)

    max_v = np.max(rate_rate)
    t1 = t[rr_l.index(max_v)]

    rate = gompertz_rates(t,k,a,b)
    r_l = list(rate)
    max_v = np.max(rate)
    t2 = t[r_l.index(max_v)]

    return t1, t2

def fit_data(data,param1,param2):
    x, y = [], []
    for s in data.keys():
        x += data[s][param1].to_list()
        y += data[s][param2].to_list()

    result = st.linregress(x,y,alternative='greater')

    return result
