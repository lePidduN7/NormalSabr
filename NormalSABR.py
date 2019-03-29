import unittest
from typing import Callable, List, Union, Dict
import numpy as np
from numba import jit, vectorize, float64 
import math
import QuantLib as ql

@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def alpha_root(atm_forward_rate, atm_forward_vol, 
                time_to_maturity, 
                beta, nu, rho):
    """
    Solver for alpha parameter as function of beta, nu, rho. Returns the smallest
    positive roots of the alpha equation such that the at-the-money basis point
    volatility is recovered exactly.
    """
    p2 = 0.25*(beta*rho*nu*time_to_maturity)/((atm_forward_rate**(1-beta))*atm_forward_vol)
    p2 -= beta*(2 - beta)*time_to_maturity/(24*(atm_forward_rate**(2 - beta)))

    p1 = (1/atm_forward_vol)*(1 + (2 - 3*rho*rho)*nu*nu*time_to_maturity/24)

    p0 = -(1/(atm_forward_rate**beta))

    coeffs = np.array([p2, p1, p0])
    roots = np.roots(coeffs)
    
    return np.where(roots <= 0, 1e5, roots).min()

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64)])
def normal_volatility(forward_rate, atm_forward_vol,
                        strike,
                        time_to_maturity,
                        alpha, beta, nu, rho):
    """
    Computes the inverse of the basis point volatility as in Hagan et al. 
    approximation, following the implementations in Bloomberg VCUB.
    Is robust for ATM neighboorhood.
    """
    factor_3_numerator = beta*(2 - beta)*alpha*alpha*time_to_maturity
    factor_3_numerator /= 24*((forward_rate*strike)**(1 - beta))

    factor_3_denominator_1 = beta*rho*alpha*nu*time_to_maturity
    factor_3_denominator_1 /= 4*((forward_rate*strike)**(0.5*(1 - beta)))

    factor_3_denominator_2 = ((2 - 3*rho*rho)*nu*nu*time_to_maturity)/24

    factor_3 = (1 + factor_3_numerator) / (1 + factor_3_denominator_1 + factor_3_denominator_2)

    if forward_rate == strike:
        factor_0 = 1/(alpha*(forward_rate**beta))
        factor_1 = 1.0
        factor_2 = 1.0
    elif np.isclose(forward_rate, strike, 1e-4, 1e-4):
        y = math.log(forward_rate/strike)
        zeta = nu*(forward_rate**(1 - beta) - strike**(1 - beta))/(alpha*(1 - beta))

        factor_0 = 1 / alpha

        factor_1_numerator = 1 + ((1 - beta)**2)*y*y/24
        factor_1_denominator = 1 + y*y/24
        factor_1 = factor_1_numerator/(factor_1_denominator * (forward_rate*strike)**(beta*0.5))

        factor_2 = 1 + 0.5*rho*zeta - ((1 - 3*rho*rho)/6)*zeta*zeta
    
    else:
        zeta = nu*(forward_rate**(1 - beta) - strike**(1 - beta))/(alpha*(1 - beta))
        x_zeta = math.log((math.sqrt(1 - 2*rho*zeta + zeta*zeta) + zeta - rho) / (1 - rho))

        factor_0 = 1 / alpha

        factor_1 = forward_rate**(1 - beta) - strike**(1 - beta)
        factor_1 /= (1 - beta)*(forward_rate - strike)

        factor_2 = x_zeta / zeta

    return factor_0 * factor_1 * factor_2 * factor_3


class ShiftedSABRSmile:
    """
    This class represents a single maturity smile for an underlying
    forward rate. The class implements the basis point volatility
    approximation from Hagan et al. and follows the implementation 
    of VCUB function from Bloomberg.
    """
    shifted_forward_rate: float
    atm_forward_volatility: float
    shift: float
    beta: float
    nu: float
    rho: float
    time_to_maturity: float
    maturity_date: ql.Date
    reference_date: ql.Date
    day_counter: ql.DayCounter

    def __init__(self, forward_rate: float, atm_forward_volatility: float,
                    beta: float, nu: float, rho: float,
                    reference_date: ql.Date, maturity_date: ql.Date,
                    shift: float = 0.0,
                    day_counter: ql.DayCounter = ql.Actual365Fixed()) -> None:
        self.shift = shift
        if self.shift < 0.0:
            raise ValueError('Shift must be non-negative')
        self.shifted_forward_rate = forward_rate + self.shift
        if self.shifted_forward_rate < 0.0:
            raise ValueError('Shifted forward rate cannot be negative')
        self.atm_forward_volatility = atm_forward_volatility
        self.reference_date = reference_date
        self.maturity_date = maturity_date
        self.day_counter = day_counter
        self.time_to_maturity = self.compute_time_to_maturity()
        self.beta = beta
        self.nu = nu
        self.rho = rho
        self.alpha = self.update_alpha()

    def compute_time_to_maturity(self) -> float:
        """
        Compute time to maturity of the smile according to the 
        day counting convention.
        """
        t0 = self.reference_date
        t1 = self.maturity_date
        return self.day_counter.yearFraction(t0, t1)
    
    def update_alpha(self) -> float:
        """
        Update alpha solving the equation such that the at-the-money
        volatility is recovered exactly.
        """
        return alpha_root(self.shifted_forward_rate, self.atm_forward_volatility,
                            self.time_to_maturity,
                            self.beta, self.nu, self.rho)  
    
    def volatility(self, strike: Union[np.ndarray, List[float], float]) -> Union[np.ndarray, float]:
        """
        Compute the annualized basis point volatilty according to
        Hagan et. al approximation of SABR model.
        """
        if type(strike) == type([]):
            strike = np.array(strike)
        elif type(strike) == float:
            strike = np.array([strike])
        strike += self.shift
        if any(s < 0 for s in strike):
            raise ValueError('Shifted strike must be non-negative')
        inverse_volatility = normal_volatility(self.shifted_forward_rate, 
                                                self.atm_forward_volatility,
                                                strike, self.time_to_maturity,
                                                self.alpha, self.beta, self.nu, self.rho)
        return 1.0/inverse_volatility

    def get_forward_rate(self) -> float:
        """
        Returns the non shifted forward rate.
        """
        return self.shifted_forward_rate - self.shift
    

class ShiftedSABRSmileSurface:
    """
    This class represents a volatility surface, that is the surface
    of n-maturities and m-strikes. It's a collection of n
    ShiftedSABRSmile objects, each one referring to one of the 
    maturities.
    """
    shifted_forward_rates: np.ndarray
    atm_forward_volatilities: np.ndarray
    shifts: np.ndarray
    betas: np.ndarray
    nu: np.ndarray
    rho: np.ndarray
    times_to_maturities: np.ndarray
    maturity_dates: np.ndarray
    reference_date: ql.Date
    day_counter: ql.DayCounter
    curves_dictionary = Dict[ql.Date, ShiftedSABRSmile]

    def __init__(self, forward_rates: List[float], 
                    atm_forward_volatilities: List[float],
                    beta: List[float],
                    nu: List[float],
                    rho: List[float],
                    reference_date: ql.Date,
                    maturity_dates: List[ql.Date],
                    shifts: Union[float, List[float]]=0.0,
                    day_counter: ql.DayCounter=ql.Actual365Fixed()) -> None:
        n_forward_rates = len(forward_rates)
        n_atm_forward_volatilities = len(atm_forward_volatilities)
        n_betas = len(beta)
        n_nu = len(nu)
        n_rho = len(rho)
        n_maturity_dates = len(maturity_dates)
        if type(shifts) == type([]):
            n_shifts = len(shifts)
        else:
            shifts = [shifts for _ in range(n_forward_rates)]
            n_shifts = n_forward_rates
        if not all(v == n_forward_rates for v in [n_atm_forward_volatilities, 
                                                    n_betas, n_nu, n_rho, 
                                                    n_maturity_dates, n_shifts]):
            raise ValueError('Dimensions of parameters should match')
        self.shifts = np.array(shifts)
        self.shifted_forward_rates = np.array(forward_rates) + shifts
        self.atm_forward_volatilities = np.array(atm_forward_volatilities)
        self.betas = np.array(beta)
        self.nus = np.array(nu)
        self.rhos = np.array(rho)
        self.reference_date = reference_date
        self.maturity_dates = np.array(maturity_dates)
        self.day_counter = day_counter
        self.curves_dictionary = self.create_curves_dictionary()
    
    def create_curves_dictionary(self):
        input_zip = zip(self.shifted_forward_rates, self.atm_forward_volatilities,
                        self.betas, self.nus, self.rhos, self.maturity_dates, 
                        self.shifts)
        curves_dictionary = dict.fromkeys(self.maturity_dates)
        for fwd, atm_vol, beta, nu, rho, maturity, shift in input_zip:
            curves_dictionary[maturity] = ShiftedSABRSmile(fwd, atm_vol,
                                                                beta, nu, rho, 
                                                                self.reference_date, maturity, 
                                                                shift, self.day_counter)
        return curves_dictionary
    


##############################################################
##############################################################
##############################################################


class ShiftedSABRSurfaceTesting(unittest.TestCase):
    def setUp(self):
        strikes_array = list(np.linspace(0.01, 0.05, num=25))
        forwards_array = list(np.linspace(0.01, 0.05, num=25))
        atm_vols_array = list(np.linspace(0.0015, 0.0045, num=25))
        shifts = list(np.ones(25) * 0.03)
        betas = list(np.ones(25) * 0.6)
        nus = list(np.random.rand(25))
        rhos = list(np.random.rand(25) - np.random.rand(25))
        reference_date = ql.Date(29, 3, 2019)
        referece_calendar = ql.TARGET()
        maturity_dates = [referece_calendar.advance(reference_date, y, ql.Years) for y in range(1, 26)]
        self.DummySurface = ShiftedSABRSmileSurface(forwards_array, atm_vols_array,
                                                    betas, nus, rhos, reference_date,
                                                    maturity_dates, shifts)

    def test_SabrSurface_istantiated(self):
        self.assertEqual(type(self.DummySurface.curves_dictionary), type(dict()))



class VolatilityFunctionsTesting(unittest.TestCase):
    def setUp(self):
        self.forward_rate = 0.02
        self.strike_rate = 0.03
        self.atm_volatility = 0.0025
        self.negative_forward_rate = - self.forward_rate
        self.negative_strike = - self.strike_rate
        self.time_to_maturity = 1.25
        self.shift = 0.02
        self.times_array = np.linspace(0.25, 20)
        self.strikes_array = np.linspace(0.01, 0.05)
        self.forwards_array = np.linspace(0.01, 0.05)
        self.atm_vols_array = np.linspace(0.0015, 0.0045)

    def test_alpha_root_is_non_negative(self):
        fwd = self.forward_rate
        atm_fwd_vol = self.atm_volatility
        ttm = self.time_to_maturity
        beta, nu, rho = 0.5, 0.45, -0.65
        self.assertGreaterEqual(alpha_root(fwd, atm_fwd_vol, ttm, beta, nu, rho), 0)

    def test_vectorized_alpha_root_is_non_negative(self):
        fwds = self.forwards_array
        atm_fwd_vols = self.atm_vols_array
        ttms = self.times_array
        beta, nu, rho = 0.5, 0.45, -0.65
        alphas = alpha_root(fwds, atm_fwd_vols, ttms, beta, nu, rho)
        self.assertTrue(all(alphas>=0))

    def test_atm_strike_volatility_is_close_to_atm_volatility(self):
        fwd = self.forward_rate
        stk = self.forward_rate
        atm_fwd_vol = self.atm_volatility
        ttm = self.time_to_maturity
        beta, nu, rho = 0.5, 0.45, -0.65
        alpha = alpha_root(fwd, atm_fwd_vol, ttm, beta, nu, rho)
        basis_point_vol = 1/normal_volatility(fwd, atm_fwd_vol, stk, ttm, alpha, beta, nu, rho)
        self.assertAlmostEqual(basis_point_vol, atm_fwd_vol)
 
    def test_vectorized_atm_strike_volatility_is_close_to_atm_volatility(self):
        fwds = self.forwards_array
        stks = self.forwards_array
        atm_fwd_vols = self.atm_vols_array
        ttm = self.times_array
        beta, nu, rho = 0.5, 0.45, -0.65
        alphas = alpha_root(fwds, atm_fwd_vols, ttm, beta, nu, rho)
        basis_point_vols = 1/normal_volatility(fwds, atm_fwd_vols, stks, ttm, alphas, beta, nu, rho)
        absolute_differences = np.abs(atm_fwd_vols - basis_point_vols)
        max_absolute_difference = np.max(absolute_differences)
        self.assertAlmostEqual(max_absolute_difference, max_absolute_difference)

class ShiftedSABRSmileTesting(unittest.TestCase):
    def setUp(self):
        self.times_array = np.linspace(0.25, 20)
        self.strikes_array = np.linspace(-0.02, 0.05)
        self.forwards_array = np.linspace(-0.02, 0.05)
        self.atm_vols_array = np.linspace(0.0015, 0.0045)
        self.shift = 0.03

        dummy_forward = 0.02
        dummy_atm_volatility = 0.003
        dummy_shift = 0.03
        dummy_beta, dummy_nu, dummy_rho = 0.6, 0.45, -0.55
        reference_date = ql.Date(29, 3, 2019)
        maturity_date = ql.Date(28, 3, 2021)

        self.DummySmile = ShiftedSABRSmile(dummy_forward, dummy_atm_volatility, 
                                            dummy_beta, dummy_nu, dummy_rho,
                                            reference_date, maturity_date,
                                            shift=dummy_shift)

    def test_SabrSmile_non_negative_alpha(self):
        self.assertGreaterEqual(self.DummySmile.alpha, 0)

    def test_SabrSmile_non_negative_time_to_maturity(self):
        self.assertGreaterEqual(self.DummySmile.time_to_maturity, 0)
    
    def test_SabrSmile_raises_exception_for_negative_shift(self):
        dummy_forward = 0.02
        dummy_atm_volatility = 0.003
        dummy_shift = -0.03
        dummy_beta, dummy_nu, dummy_rho = 0.6, 0.45, -0.55
        reference_date = ql.Date(29, 3, 2019)
        maturity_date = ql.Date(28, 3, 2021)
        with self.assertRaises(ValueError):
            ShiftedSABRSmile(dummy_forward, dummy_atm_volatility, 
                                dummy_beta, dummy_nu, dummy_rho,
                                reference_date, maturity_date,
                                shift=dummy_shift)

    def test_SabrSmile_raises_exception_for_negative_shifted_forward_rate(self):
        dummy_forward = -0.02
        dummy_atm_volatility = 0.003
        dummy_shift = 0.015
        dummy_beta, dummy_nu, dummy_rho = 0.6, 0.45, -0.55
        reference_date = ql.Date(29, 3, 2019)
        maturity_date = ql.Date(28, 3, 2021)
        with self.assertRaises(ValueError):
            ShiftedSABRSmile(dummy_forward, dummy_atm_volatility, 
                                dummy_beta, dummy_nu, dummy_rho,
                                reference_date, maturity_date,
                                shift=dummy_shift)

    def test_SabrSmile_volatility_is_non_negative(self):
        strike = 0.01
        self.assertGreaterEqual(self.DummySmile.volatility(strike), 0)

    def test_SabrSmile_volatility_curve_is_non_negative(self):
        forward_rate = self.DummySmile.get_forward_rate()
        strikes = np.linspace(-0.015, 0.015) + forward_rate
        smile_curve = self.DummySmile.volatility(strikes)
        self.assertTrue(all(v > 0 for v in smile_curve))
    
    def test_SabrSmile_volatility_returns_array_of_vols_for_array_of_strikes(self):
        forward_rate = self.DummySmile.get_forward_rate()
        strikes = np.linspace(-0.015, 0.015) + forward_rate
        smile_curve = self.DummySmile.volatility(strikes)
        self.assertEqual(strikes.size, smile_curve.size)

    def test_SabrSmile_raises_exception_for_negative_shifted_strike(self):
        strike = -0.05
        with self.assertRaises(ValueError):
            self.DummySmile.volatility(strike)
    
    def test_SabrSmile_raises_exception_if_any_shifted_strike_is_negative(self):
        forward_rate = self.DummySmile.get_forward_rate()
        strikes = np.linspace(-0.08, 0.015) + forward_rate
        with self.assertRaises(ValueError):
            self.DummySmile.volatility(strikes)

    def test_SabrSmile_volatility_is_stable_around_atm(self):
        otm_strike_offset = 1e-6
        strike_plus = self.DummySmile.get_forward_rate() + otm_strike_offset
        strike_minus = self.DummySmile.get_forward_rate() - otm_strike_offset
        otm_volatility_plus = self.DummySmile.volatility(strike_plus)
        otm_volatility_minus = self.DummySmile.volatility(strike_minus)
        volatility_absolute_difference = abs(otm_volatility_plus - otm_volatility_minus)
        self.assertGreaterEqual(2*otm_strike_offset, volatility_absolute_difference)

        
if __name__ == "__main__":
    unittest.main(verbosity=2)