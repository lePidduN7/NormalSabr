from datetime import date
from dataclasses import dataclass, field
from typing import Callable, List, Union, Dict
import numpy as np
from QuantLib import Date, DayCounter, Actual360, Actual365Fixed, ActualActual

from numerix import alpha_root, normal_volatility

@dataclass
class ShiftedSABRSmile:
    """
    This class represents a single maturity smile for an underlying
    forward rate. The class implements the basis point volatility
    approximation from Hagan et al. and follows the implementation 
    of VCUB function from Bloomberg.
    """
    forward_rate: float
    atm_forward_volatility: float
    beta: float
    nu: float
    rho: float
    reference_date: date
    maturity_date: date
    shift: float = 0.0
    day_counting: str = 'ACT/ACT'
    shifted_forward_rate: float = field(init=False)
    time_to_maturity: float = field(init=False)
    alpha: float = field(init=False)
    day_counter: DayCounter = field(init=False)

    def __post_init__(self) -> None:
        if self.shift < 0.0:
            raise ValueError('Shift must be non-negative')
        self.shifted_forward_rate = self.forward_rate + self.shift
        if self.shifted_forward_rate <= 0.0: 
            raise ValueError('Shifted forward rate must be positive')
        if (self.beta > (1 - 1e-4)) or (self.beta < 1e-4):
            raise ValueError('Unstable values of beta parameter')
        if abs(self.rho) > 1.0:
            raise ValueError('Incorrect value of rho parameter')
        if self.maturity_date <= self.reference_date:
            raise ValueError('Maturity cannot be <= than reference date')
        
        self.day_counter = self.set_day_counter()        
        self.time_to_maturity = self.compute_time_to_maturity()
        self.alpha = self.update_alpha()
        if self.alpha <= 0:
            raise ValueError('Alpha is negative: check input values')

    def set_day_counter(self, convention: str = 'ACT/ACT') -> DayCounter:
        """
        Set DayCounter object based on day counting convention
        expressed at instantiation.
        """
        self.day_counting = convention.upper()
        if self.day_counting == 'ACT/ACT':
            return ActualActual()
        elif self.day_counting == 'ACT/365':
            return Actual365Fixed()
        elif self.day_counting == 'ACT/360':
            return Actual360()
        else:
            raise NotImplementedError('Day counting convention no implemented')

    def compute_time_to_maturity(self) -> float:
        """
        Compute time to maturity of the smile according to the 
        day counting convention.
        """
        t0 = Date.from_date(self.reference_date)
        t1 = Date.from_date(self.maturity_date)
        return self.day_counter.yearFraction(t0, t1)
    
    def update_alpha(self) -> float:
        """
        Update alpha solving the equation such that the at-the-money
        volatility is recovered exactly.
        """
        return alpha_root(self.shifted_forward_rate, self.atm_forward_volatility,
                            self.time_to_maturity,
                            self.beta, self.nu, self.rho)  
    
    def update_at_the_money(self, new_forward: float, new_atm_volatility: float) -> None:
        """
        Update the smile with real time data and recompute alpha
        """
        self.forward_rate = new_forward
        self.shifted_forward_rate = new_forward + self.shift
        self.atm_forward_volatility = new_atm_volatility
        self.alpha = self.update_alpha()
    
    def volatility(self, strike: float) -> float:
        """
        Compute the annualized basis point volatilty according to
        Hagan et. al approximation of SABR model.
        """
        strike += self.shift
        if strike <= 0:
            raise ValueError('Shifted strike must be positive')
        inverse_volatility = normal_volatility(self.shifted_forward_rate, 
                                                self.atm_forward_volatility,
                                                strike, self.time_to_maturity,
                                                self.alpha, self.beta, self.nu, self.rho)
        return 1.0/inverse_volatility
    
    def volatility_curve(self, strikes: List[float]) -> List[float]:
        """
        Computes the volatility curve for a set of strikes
        """
        strikes = [s + self.shift for s in strikes]
        if any(s <= 0 for s in strikes):
            raise ValueError('Shifted strikes must be positive')
        inverse_volatility = normal_volatility(self.shifted_forward_rate, 
                                                self.atm_forward_volatility,
                                                strikes, self.time_to_maturity,
                                                self.alpha, self.beta, self.nu, self.rho)
        return list(1.0/inverse_volatility)
 
# class ShiftedSABRSmileSurface:
#     """
#     This class represents a volatility surface, that is the surface
#     of n-maturities and m-strikes. It's a collection of n
#     ShiftedSABRSmile objects, each one referring to one of the 
#     maturities.
#     """
#     shifted_forward_rates: np.ndarray
#     atm_forward_volatilities: np.ndarray
#     shifts: np.ndarray
#     betas: np.ndarray
#     nu: np.ndarray
#     rho: np.ndarray
#     times_to_maturities: np.ndarray
#     maturity_dates: np.ndarray
#     reference_date: ql.Date
#     day_counter: ql.DayCounter
#     curves_dictionary = Dict[ql.Date, ShiftedSABRSmile]

#     def __init__(self, forward_rates: List[float], 
#                     atm_forward_volatilities: List[float],
#                     beta: List[float],
#                     nu: List[float],
#                     rho: List[float],
#                     reference_date: ql.Date,
#                     maturity_dates: List[ql.Date],
#                     shifts: Union[float, List[float]]=0.0,
#                     day_counter: ql.DayCounter=ql.Actual365Fixed()) -> None:
#         n_forward_rates = len(forward_rates)
#         n_atm_forward_volatilities = len(atm_forward_volatilities)
#         n_betas = len(beta)
#         n_nu = len(nu)
#         n_rho = len(rho)
#         n_maturity_dates = len(maturity_dates)
#         if type(shifts) == type([]):
#             n_shifts = len(shifts)
#         else:
#             shifts = [shifts for _ in range(n_forward_rates)]
#             n_shifts = n_forward_rates
#         if not all(v == n_forward_rates for v in [n_atm_forward_volatilities, 
#                                                     n_betas, n_nu, n_rho, 
#                                                     n_maturity_dates, n_shifts]):
#             raise ValueError('Dimensions of parameters should match')
#         self.shifts = np.array(shifts)
#         self.shifted_forward_rates = np.array(forward_rates) + shifts
#         self.atm_forward_volatilities = np.array(atm_forward_volatilities)
#         self.betas = np.array(beta)
#         self.nus = np.array(nu)
#         self.rhos = np.array(rho)
#         self.reference_date = reference_date
#         self.maturity_dates = np.array(maturity_dates)
#         self.day_counter = day_counter
#         self.curves_dictionary = self.create_curves_dictionary()
    
#     def create_curves_dictionary(self) -> Dict[ql.Date, ShiftedSABRSmile]:
#         """
#         Instantiates the volatility curves for every maturity
#         and laod them into a dictionary indexed by maturity dates
#         """
#         input_zip = zip(self.shifted_forward_rates, self.atm_forward_volatilities,
#                         self.betas, self.nus, self.rhos, self.maturity_dates, 
#                         self.shifts)
#         curves_dictionary = dict.fromkeys(self.maturity_dates)
#         for fwd, atm_vol, beta, nu, rho, maturity, shift in input_zip:
#             curves_dictionary[maturity] = ShiftedSABRSmile(fwd, atm_vol,
#                                                                 beta, nu, rho, 
#                                                                 self.reference_date, maturity, 
#                                                                 shift, self.day_counter)
#         return curves_dictionary
    
#     def make_strikes_matrix(self, offset_array: List[float]):
#         """
#         Creates an n-forwards x m-offsets matrix of strikes. Every strike is obtained 
#         by adding the offset to the forward rate. The offset may or may not contain
#         the zero and the result is going to be identical since alpha parameter is 
#         obtained by recovering exactly the atm volatility.
#         """
#         return (np.atleast_2d(self.get_forward_rates()).T + np.array(offset_array)).tolist()


#     def get_maturity_dates(self):
#         return self.maturity_dates

#     def get_forward_rates(self):
#         return self.shifted_forward_rates - self.shifts
    


##############################################################
##############################################################
##############################################################


# class ShiftedSABRSurfaceTesting(unittest.TestCase):
#     def setUp(self):
#         strikes_array = list(np.linspace(0.01, 0.05, num=25))
#         forwards_array = list(np.linspace(0.01, 0.05, num=25))
#         atm_vols_array = list(np.linspace(0.0015, 0.0045, num=25))
#         shifts = list(np.ones(25) * 0.03)
#         betas = list(np.ones(25) * 0.6)
#         nus = list(np.random.rand(25))
#         rhos = list(np.random.rand(25) - np.random.rand(25))
#         reference_date = ql.Date(29, 3, 2019)
#         referece_calendar = ql.TARGET()
#         maturity_dates = [referece_calendar.advance(reference_date, y, ql.Years) for y in range(1, 26)]
#         self.DummySurface = ShiftedSABRSmileSurface(forwards_array, atm_vols_array,
#                                                     betas, nus, rhos, reference_date,
#                                                     maturity_dates, shifts)

#     def test_SabrSurface_istantiated(self):
#         self.assertEqual(type(self.DummySurface.curves_dictionary), type(dict()))

#     def test_SabrSurface_creates_strikes_surface(self):
#         offsets = np.arange(-0.02, 0.025, 0.005)
#         forward_rates = self.DummySurface.get_forward_rates()
#         strikes_matrix = self.DummySurface.make_strikes_matrix(offsets)
#         strikes_matrix_shape = (len(strikes_matrix), len(strikes_matrix[0]))
#         n_rows = forward_rates.size
#         n_columns = offsets.size
#         self.assertEqual(strikes_matrix_shape, (n_rows, n_columns))

