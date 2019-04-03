import pytest
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from lmfit import Parameters, minimize

import QuantLib as ql

from NormalSABR import ShiftedSABRSmile, ShiftedSABRSmileSurfaceCalibration
from NormalSABR import volatility_curve_fit_objective_function, volatility_surface_fit_objective_function

@dataclass
class InitData_curve:
    forward_rate: float = 0.02
    atm_volatility: float = 0.003
    shift: float = 0.03
    beta: float = 0.6
    nu: float = 0.45
    rho: float = -0.55 
    reference_date: date = date(2019, 3, 28)
    maturity_date: date = date(2021, 3, 28)
    day_counter = 'ACT/360'

    def get_sabr_parameters(self):
        return self.beta, self.nu, self.rho

@dataclass
class InitData_surface:
    forward_rates: List[float]
    atm_volatilities: List[float]
    shift: float
    beta: List[float]
    nu: List[float]
    rho: List[float]
    reference_date: date
    maturities: List[date]

@pytest.fixture
def basic_setup():
    return InitData_curve()

@pytest.fixture
def advanced_setup():
    forward_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
    atm_volatilities = [0.003,0.003,0.003,0.003,0.003] 
    shift = 0.03
    beta = [0.6, 0.6, 0.6, 0.6, 0.6]
    nu = [0.25,0.25,0.25,0.25,0.25]
    rho = [0.0, 0.0, 0.0, 0.0, 0.0]
    reference_date = date(2019, 3, 28)
    maturities = [date(2020, 3, 28), date(2021, 3, 28), date(2022, 3, 28),
                    date(2023, 3, 28), date(2024, 3, 28)]
    init_data_surface = InitData_surface(forward_rates, 
                                            atm_volatilities,
                                            shift,
                                            beta, nu, rho,
                                            reference_date,
                                            maturities)
    return init_data_surface

def advanced_setup_direct():
    forward_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
    atm_volatilities = [0.003,0.003,0.003,0.003,0.003] 
    shift = 0.03
    beta = [0.6, 0.6, 0.6, 0.6, 0.6]
    nu = [0.25,0.25,0.25,0.25,0.25]
    rho = [0.0, 0.0, 0.0, 0.0, 0.0]
    reference_date = date(2019, 3, 28)
    maturities = [date(2020, 3, 28), date(2021, 3, 28), date(2022, 3, 28),
                    date(2023, 3, 28), date(2024, 3, 28)]
    init_data_surface = InitData_surface(forward_rates, 
                                            atm_volatilities,
                                            shift,
                                            beta, nu, rho,
                                            reference_date,
                                            maturities)
    return init_data_surface

def test_surface_instantiation_correct(advanced_setup):
    fwds = advanced_setup.forward_rates
    atm_vols = advanced_setup.atm_volatilities
    shift = advanced_setup.shift
    ref_date = advanced_setup.reference_date
    maturity_dates = advanced_setup.maturities
    DummySurface = ShiftedSABRSmileSurfaceCalibration(ref_date,
                                                        maturity_dates,
                                                        fwds,
                                                        atm_vols,
                                                        shift)

def test_surface_set_parameters_dictionary(advanced_setup):
    fwds = advanced_setup.forward_rates
    atm_vols = advanced_setup.atm_volatilities
    shift = advanced_setup.shift
    ref_date = advanced_setup.reference_date
    maturity_dates = advanced_setup.maturities
    DummySurface = ShiftedSABRSmileSurfaceCalibration(ref_date,
                                                        maturity_dates,
                                                        fwds,
                                                        atm_vols,
                                                        shift)
    DummySurface.set_parameters_dict(False, False, True)
    DummySurface.parameters_dict.pretty_print()
   
def test_curve_instatiation_correct(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    ref_date = basic_setup.reference_date
    end_date = basic_setup.maturity_date
    dc = basic_setup.day_counter
    shift = basic_setup.shift
    DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                    ref_date, end_date,
                                    shift, dc)

def test_update_at_the_money_returns_new_atm_vol(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    ref_date = basic_setup.reference_date
    end_date = basic_setup.maturity_date
    dc = basic_setup.day_counter
    shift = basic_setup.shift
    DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                    ref_date, end_date,
                                    shift, dc)
    v_0 = DummySmile.volatility(fwd)
    assert abs(v_0 - atm_vol) <= 1e-6
    fwd += 0.01
    atm_vol += 0.0003
    DummySmile.update_at_the_money(fwd, atm_vol)
    v_1 = DummySmile.volatility(fwd)
    assert abs(v_1 - atm_vol) <= 1e-6

def test_raise_ValueError_if_negative_shifted_forward(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    ref_date = basic_setup.reference_date
    end_date = basic_setup.maturity_date
    dc = basic_setup.day_counter
    shift = -1 * basic_setup.shift
    with pytest.raises(ValueError):
        DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                        ref_date, end_date,
                                        shift, dc)

def test_raise_ValueError_if_maturity_is_less_than_ref_date(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    ref_date = basic_setup.reference_date
    end_date = ref_date - timedelta(10)
    dc = basic_setup.day_counter
    shift = basic_setup.shift
    with pytest.raises(ValueError):
        DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                        ref_date, end_date,
                                        shift, dc)

def test_raise_ValueError_if_beta_has_incorrect_values(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    b = -0.2
    ref_date = basic_setup.reference_date
    end_date = basic_setup.maturity_date
    dc = basic_setup.day_counter
    shift = basic_setup.shift
    with pytest.raises(ValueError):
        DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                        ref_date, end_date,
                                        shift, dc)

def test_raise_ValueError_if_rho_has_incorrect_values(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    p = - 1.2
    ref_date = basic_setup.reference_date
    end_date = basic_setup.maturity_date
    dc = basic_setup.day_counter
    shift = basic_setup.shift
    with pytest.raises(ValueError):
        DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                        ref_date, end_date,
                                        shift, dc)

def test_volatility_is_non_negative(basic_setup):
    fwd = basic_setup.forward_rate
    atm_vol = basic_setup.atm_volatility
    b, v, p = basic_setup.get_sabr_parameters()
    ref_date = basic_setup.reference_date
    end_date = basic_setup.maturity_date
    dc = basic_setup.day_counter
    shift = basic_setup.shift
    DummySmile = ShiftedSABRSmile(fwd, atm_vol, b, v, p,
                                    ref_date, end_date,
                                    shift, dc)
    strikes = [fwd - i*0.01 for i in range(3)] + [fwd + i*0.01 for i in range(3)]
    volatility_curve = DummySmile.volatility_curve(strikes)
    assert all(v >= 0 for v in volatility_curve)

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
    forward_rates = [0.0236, 0.0235, 0.0267, 0.0290, 0.0277]
    atm_volatilities = [0.00573, 0.00596, 0.0065, 0.006462, 0.00543]
    reference_date =  date(2019, 4, 3)
    maturity_dates = [date(2019, 7, 3), date(2020, 4, 3), date(2024, 4, 3),
                        date(2029, 4, 3), date(2039, 4, 3)]


    true_vols = [ [ 0.00737, 0.00642, 0.0060, 0.00567, 0.00593, 0.0069 ],
                            [ 0.0073, 0.00653, 0.0062, 0.0059, 0.0060, 0.0067 ],
                            [ 0.00693, 0.00667, 0.00656, 0.00647, 0.00651, 0.00676 ],
                            [ 0.00681, 0.0066, 0.0065, 0.00644, 0.00646, 0.00663 ],
                            [ 0.00588, 0.005625, 0.0055, 0.005378, 0.00538, 0.00555 ],
    ]
    offsets = [-100, -50, -25, 25, 50, 100]

    volatility_spreads = [ [(true_vol - atm_volatility)*1e4 for true_vol in true_vols[ix]]
                            for ix, atm_volatility in enumerate(atm_volatilities) ]

    DummySurface = ShiftedSABRSmileSurfaceCalibration(reference_date, maturity_dates, 
                                                        forward_rates, atm_volatilities,
                                                        shift=0.0)
    init_time = datetime.now()
    opt_out = DummySurface.calibrate(offsets, volatility_spreads)
    print(datetime.now() - init_time)
    opt_out.params.pretty_print()


if __name__ == "__main__":
    # test_surface_set_parameters_dictionary(advanced_setup_direct())
    # curve_fitting()
    surface_fitting()
    
