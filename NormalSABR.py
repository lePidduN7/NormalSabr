from datetime import date
from dataclasses import dataclass, field
from typing import Callable, List, Union, Dict
import numpy as np
from QuantLib import Date, DayCounter, Actual360, Actual365Fixed, ActualActual

from numerix import alpha_root, normal_volatility, normal_volatility_surface
from lmfit import Parameters, minimize

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
    day_counting: str = 'ACT/365'
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
    
    def __lt__(self, other):
        if self.maturity_date < other.maturity_date:
            return True
        else:
            return False
    
    def __le__(self, other):
        if self.maturity_date <= other.maturity_date:
            return True
        else:
            return False
    
    def __gt__(self, other):
        if self.maturity_date > other.maturity_date:
            return True
        else: 
            return False
    
    def __ge__(self, other):
        if self.maturity_date >= other.maturity_date:
            return True
        else:
            return False
    
    def __eq__(self, other):
        if self.maturity_date == other.maturity_date:
            return True
        else:
            return False

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
 
@dataclass
class ShiftedSABRSmileSurfaceCalibration:
    """
    This class represents the volatility surface for a given underlying.
    It takes care of collecting the different quoted smiles as input and 
    fit a parametrized surface to the normal implied volatilities.
    """
    reference_date: date
    maturities: List[date]
    atm_forwards: List[float]
    atm_volatilities: List[float]
    shift: float
    curves_dict: Dict[date, ShiftedSABRSmile] = field(init=False)
    parameters_dict: Parameters = field(init=False)
    calibrated_parameters_dict: Parameters = field(init=False)

    def __post_init__(self):
        self.curves_dict = dict.fromkeys(self.maturities)
        for date, forward_rate, atm_volatility in zip(self.maturities,
                                                        self.atm_forwards,
                                                        self.atm_volatilities):
            sabr_vol_curve = ShiftedSABRSmile(forward_rate, atm_volatility, 0.89, 0.25, 0.0,
                                                self.reference_date, date, self.shift)
            self.curves_dict.update({date: sabr_vol_curve})
    
    def set_parameters_dict(self, is_beta_fixed: bool=False,
                            is_nu_fixed: bool=False, 
                            is_rho_fixed: bool=True) -> None:
        """
        This function set the initial Parameters dictionary
        for the lmfit procedure
        """
        self.parameters_dict = Parameters()
        min_beta = 1e-4
        max_beta = 1 - min_beta
        for d in self.maturities:
            self.parameters_dict.add('beta{:{dfmt}}'.format(d, dfmt='%Y%m%d'), value=0.89, 
                                    min=min_beta, max=max_beta,
                                    vary=not is_beta_fixed)
            self.parameters_dict.add('nu{:{dfmt}}'.format(d, dfmt='%Y%m%d'), value=0.1, 
                                    min=min_beta, 
                                    vary=not is_nu_fixed)
            self.parameters_dict.add('rho{:{dfmt}}'.format(d, dfmt='%Y%m%d'), value=0.0,
                                    min= -1, max=1, 
                                    vary=not is_rho_fixed)
    
    def calibrate(self, atm_strike_offsets: List[float], 
                    atm_vols_spreads_matrix: List[List[float]],
                    is_beta_fixed: bool=False,
                    is_nu_fixed: bool=False,
                    is_rho_fixed: bool=True) -> None:
        volatility_smiles = sorted(self.curves_dict.values())
        self.set_parameters_dict(is_beta_fixed, is_nu_fixed, is_rho_fixed)
        shifted_forwards = [smile.shifted_forward_rate for smile in volatility_smiles]
        atm_volatilities = [smile.atm_forward_volatility for smile in volatility_smiles]
        time_to_maturities = [smile.time_to_maturity for smile in volatility_smiles]
        strikes_matrix = [[shifted_forward + offset*1e-4 for offset in atm_strike_offsets] 
                            for shifted_forward in shifted_forwards]
        target_volatilities = [[smile.atm_forward_volatility + vol_spread*1e-4 for vol_spread in atm_vols_spreads_matrix[ix]] 
                                for ix, smile in enumerate(volatility_smiles)]
        weights_matrix = [[1.0 for _ in atm_strike_offsets]
                            for _ in volatility_smiles]
        calibration_output = minimize(volatility_surface_fit_objective_function,
                                        self.parameters_dict,
                                        args=(self.maturities,
                                                shifted_forwards,
                                                atm_volatilities,
                                                time_to_maturities,
                                                strikes_matrix,
                                                target_volatilities,
                                                weights_matrix))
        return calibration_output

def volatility_curve_fit_objective_function(params: Parameters,
                                            curve_object: ShiftedSABRSmile,
                                            strikes_vector: List[float],
                                            target_volatilities: List[float],
                                            weights: List[float]) -> List[float]:
    forward_rate = ShiftedSABRSmile.shifted_forward_rate
    atm_volatility = ShiftedSABRSmile.atm_forward_volatility
    time_to_maturity = ShiftedSABRSmile.time_to_maturity
    params_values_dict = params.valuesdict()
    b = params_values_dict['beta']
    v = params_values_dict['nu']
    p = params_values_dict['rho']
    a = alpha_root(forward_rate, atm_volatility, time_to_maturity,
                    b, v, p)
    basis_point_vols = 1.0/normal_volatility(forward_rate, atm_volatility,
                                                strikes_vector, time_to_maturity,
                                                a, b, v, p)
    weighted_errors = list((basis_point_vols - np.array(target_volatilities))*np.array(weights))
    return weighted_errors
    
def volatility_surface_fit_objective_function(params: Parameters,
                                                maturity_dates: List[date],
                                                forward_rates: List[float],
                                                atm_volatilities: List[float],
                                                time_to_maturities: List[float],
                                                strikes_matrix: List[List[float]],
                                                target_volatilities: List[List[float]],
                                                weights: List[List[float]]) -> List[List[float]]:
    target_volatilities_array = np.array(target_volatilities)
    weights_array = np.array(weights)
    n_curves, n_strikes = target_volatilities_array.shape
    params_vals = params.valuesdict()
    b = np.fromiter((params_vals['beta{:{dfmt}}'.format(d, dfmt='%Y%m%d')] for d in maturity_dates),
                    dtype=np.float64)
    v = np.fromiter((params_vals['nu{:{dfmt}}'.format(d, dfmt='%Y%m%d')] for d in maturity_dates),
                    dtype=np.float64)
    p = np.fromiter((params_vals['rho{:{dfmt}}'.format(d, dfmt='%Y%m%d')] for d in maturity_dates),
                    dtype=np.float64)
    output_surface = np.zeros((n_curves, n_strikes))
    normal_volatility_surface(strikes_matrix, b, v, p, forward_rates,
                                atm_volatilities, time_to_maturities, output_surface)
    return ((output_surface - target_volatilities_array)*weights_array).flatten().tolist()
        
        

    
