import pytest
from dataclasses import dataclass
from typing import List
import numpy as np
from lmfit import minimize, Parameters
from numerix import normal_volatility_surface, normal_volatility, alpha_root 
from numerix import volatility_curve_fit_objective_function, volatility_surface_fit_objective_function

@dataclass
class InitData:
    forward_rate: float
    strike_rate: float
    atm_volatility: float
    negative_forward_rate: float
    negative_strike: float
    time_to_maturity: float
    shift: float
    times_array: List[float]
    strikes_array: List[float]
    forwards_array: List[float]
    atm_vols_array: List[float]


@pytest.fixture
def setUp():
    forward_rate = 0.02
    strike_rate = 0.03
    atm_volatility = 0.0025
    negative_forward_rate = - forward_rate
    negative_strike = - strike_rate
    time_to_maturity = 1.25
    shift = 0.02
    times_array = list(np.linspace(0.25, 20))
    strikes_array = list(np.linspace(0.01, 0.05))
    forwards_array = list(np.linspace(0.01, 0.05))
    atm_vols_array = list(np.linspace(0.0015, 0.0045))
    return InitData(forward_rate, strike_rate, atm_volatility,
                    negative_forward_rate, negative_strike,
                    time_to_maturity, shift, times_array,
                    strikes_array, forwards_array, atm_vols_array)

def test_alpha_root_is_non_negative(setUp):
    fwd = setUp.forward_rate
    atm_fwd_vol = setUp.atm_volatility
    ttm = setUp.time_to_maturity
    beta, nu, rho = 0.5, 0.45, -0.65
    assert alpha_root(fwd, atm_fwd_vol, ttm, beta, nu, rho) >= 0

def test_vectorized_alpha_root_is_non_negative(setUp):
    fwds = setUp.forwards_array
    atm_fwd_vols = setUp.atm_vols_array
    ttms = setUp.times_array
    beta, nu, rho = 0.5, 0.45, -0.65
    alphas = alpha_root(fwds, atm_fwd_vols, ttms, beta, nu, rho)
    assert all(alphas>=0)

def test_atm_strike_volatility_is_close_to_atm_volatility(setUp):
    fwd = setUp.forward_rate
    stk = fwd
    atm_fwd_vol = setUp.atm_volatility
    ttm = setUp.time_to_maturity
    beta, nu, rho = 0.5, 0.45, -0.65
    alpha = alpha_root(fwd, atm_fwd_vol, ttm, beta, nu, rho)
    basis_point_vol = 1/normal_volatility(fwd, atm_fwd_vol, stk, ttm, alpha, beta, nu, rho)
    absolute_error = abs(basis_point_vol - atm_fwd_vol)
    assert absolute_error <= 1e-8

def test_vectorized_atm_strike_volatility_is_close_to_atm_volatility(setUp):
    fwds = setUp.forwards_array
    stks = setUp.forwards_array
    atm_fwd_vols = setUp.atm_vols_array
    ttm = setUp.times_array
    beta, nu, rho = 0.5, 0.45, -0.65
    alphas = alpha_root(fwds, atm_fwd_vols, ttm, beta, nu, rho)
    basis_point_vols = 1.0/normal_volatility(fwds, atm_fwd_vols, stks, ttm, alphas, beta, nu, rho)
    absolute_differences = np.abs(atm_fwd_vols - basis_point_vols)
    max_absolute_difference = np.max(absolute_differences)
    assert max_absolute_difference <= 1e-8

def test_surface_output(setUp):
    fwds = [fwd + setUp.shift for fwd in setUp.forwards_array]
    atm_fwd_vols = setUp.atm_vols_array
    ttm = setUp.times_array
    bs = np.ones(50)*0.6
    ns = np.ones(50)*0.45
    rs = -np.ones(50)*0.65
    offsets = [-0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01]
    stk_matrix = np.atleast_2d(fwds).T + np.atleast_2d(offsets)
    bp_vol_surface = np.zeros(stk_matrix.shape)
    normal_volatility_surface(stk_matrix, bs, ns, rs, 
                                fwds, atm_fwd_vols, ttm,
                                bp_vol_surface)
    assert stk_matrix.shape == bp_vol_surface.shape

def test_surface_output_for_zero_offsets_is_the_atm_volatility(setUp):
    fwds = [fwd + setUp.shift for fwd in setUp.forwards_array]
    atm_fwd_vols = setUp.atm_vols_array
    ttm = setUp.times_array
    bs = np.ones(50)*0.6
    ns = np.ones(50)*0.45
    rs = -np.ones(50)*0.65
    offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    stk_matrix = np.atleast_2d(fwds).T + np.atleast_2d(offsets)
    bp_vol_surface = np.zeros(stk_matrix.shape)
    normal_volatility_surface(stk_matrix, bs, ns, rs, 
                                fwds, atm_fwd_vols, ttm,
                                bp_vol_surface)
    max_absolute_difference = max([max(abs(atm_fwd_vols[ix] - bp_vol_surface[ix])) 
                                    for ix, _ in enumerate(atm_fwd_vols)])
    assert max_absolute_difference <= 1e-8


def curve_fitting():
    min_beta = 1e-4
    max_beta = 1 - min_beta
    sabr_params = Parameters()
    sabr_params.add('beta', value=0.6, min=min_beta, max=max_beta)
    sabr_params.add('nu', value=0.45, min=min_beta)
    sabr_params.add('rho', value=0.0, vary=False)
    forward_rate = 0.02
    atm_volatility = 0.0035
    time_to_maturity = 2.5
    offsets = [-0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01]
    strikes = [forward_rate + offset for offset in offsets]
    volatility_spreads = [-9.26,- 5.32, -2.77, 2.9, 5.88, 11.89]
    target_volatilities = [atm_volatility + spread*1e-4 for spread in volatility_spreads]
    weights = [1, 1, 1, 1, 1, 1]
    opt_results = minimize(volatility_curve_fit_objective_function, sabr_params,
                            args=(forward_rate, atm_volatility, time_to_maturity, 
                                    strikes, target_volatilities, weights))
    opt_results.params.pretty_print()
    print(opt_results.residual)

def surface_fitting():
    forward_rates = [2.55, 2.50, 2.43, 2.36, 2.29, 2.23, 2.28, 2.38,
                    2.48, 2.57, 2.64, 2.71, 2.78, 2.82, 2.80, 2.71,2.66]
    forward_rates = [x*0.01 for x in forward_rates]
    n_maturities = len(forward_rates)
    atm_volatilities = [0.0036, 0.0041, 0.0050, 0.0055, 0.0058, 0.0064, 
                        0.0066, 0.0066, 0.0067, 0.0068, 0.0068, 0.0068,
                        0.0067, 0.0067, 0.0062, 0.0057, 0.0051]
    volatility_spreads = [-9.26,- 5.32, -2.77, 2.9, 5.88, 11.89]
    target_volatilities = [[atm_volatility + spread*1e-4 for spread in volatility_spreads]
                            for atm_volatility in atm_volatilities]
    offsets = [-0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01]
    strikes = [[forward_rate + offset for offset in offsets]
                for forward_rate in forward_rates]
    print(target_volatilities)
    print(strikes)
    print()

    weights = [[1e4, 1e4, 1e4, 1e4, 1e4, 1e4] for _ in range(n_maturities)]
    time_to_maturities = np.linspace(1, 20, num=n_maturities)

    min_beta = 1e-4
    max_beta = 1 - min_beta

    sabr_params = Parameters()
    for i in range(n_maturities):
        sabr_params.add('beta{}'.format(i), value=0.6, min=min_beta, max=max_beta)
        sabr_params.add('nu{}'.format(i), value=0.45, min=min_beta)
        sabr_params.add('rho{}'.format(i), value=0.0, vary=False)
    opt_results = minimize(volatility_surface_fit_objective_function, sabr_params,
                            args=(forward_rates, atm_volatilities, time_to_maturities, strikes,
                                    target_volatilities, weights))
    opt_results.params.pretty_print()
    print(opt_results.residual)

if __name__ == "__main__":
    curve_fitting()
    surface_fitting()

