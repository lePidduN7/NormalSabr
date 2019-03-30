import pytest
from dataclasses import dataclass

import QuantLib as ql

from NormalSABR import ShiftedSABRSmile

@dataclass
class InitData:
    forward_rate: float = 0.02
    atm_volatility: float = 0.003
    shift: float = 0.03
    beta: float = 0.6
    nu: float = 0.45
    rho: float = -0.55 
    reference_date: ql.Date = ql.Date(29, 3, 2019)
    maturity_date: ql.Date = ql.Date(28, 3, 2021)
    day_counter: ql.DayCounter = ql.Actual365Fixed()

    def get_sabr_parameters(self):
        return self.beta, self.nu, self.rho

@pytest.fixture
def basic_setup():
    return InitData()

def test_instatiation_correct(basic_setup):
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
    end_date = ref_date - 10
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





