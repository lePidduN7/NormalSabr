import pytest
from typing import List, Dict
from dataclasses import dataclass
from datetime import date, timedelta

import QuantLib as ql

from NormalSABR import ShiftedSABRSmile, ShiftedSABRSmileSurfaceCalibration

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



if __name__ == "__main__":
    test_surface_set_parameters_dictionary(advanced_setup_direct())
    
