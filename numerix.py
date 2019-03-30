import math
import numpy as np
from numba import jit, vectorize, guvectorize, float64

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

@guvectorize([(float64[:,:], float64[:], float64[:], float64[:], 
                float64[:], float64[:], float64[:],
                float64[:,:])], '(n,m),(n),(n),(n),(n),(n),(n)->(n,m)')
def normal_volatility_surface(strikes_matrix, betas, nus, rhos,
                                forwards, atm_vols, time_to_maturities,
                                output_surface):
    """
    Computes the volatility surface upon a given matrix of strikes and
    Sabr parameters arrays. The main purpose is for fitting the parameters
    """
    alphas = alpha_root(forwards, atm_vols, time_to_maturities,
                        betas, nus, rhos)
    for ix, strikes in enumerate(strikes_matrix):
        output_surface[ix] = 1.0/normal_volatility(forwards[ix],
                                                    atm_vols[ix],
                                                    strikes,
                                                    time_to_maturities[ix],
                                                    alphas[ix], betas[ix],
                                                    nus[ix], rhos[ix])
    
