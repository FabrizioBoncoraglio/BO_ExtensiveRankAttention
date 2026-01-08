import numpy as np
import pandas as pd
from scipy.integrate import quad
from sys import argv
from numpy.random import default_rng
from scipy.stats import multivariate_normal


rng = default_rng()


def stieltjes_root(z, sigma, rho):
    alpha = 1 / rho
    R_noise = sigma
    a3 = np.sqrt(alpha) * R_noise
    a2 = -(np.sqrt(alpha) * z + R_noise) #
    a1 = (z + np.sqrt(alpha) - alpha**(-1 / 2))
    a0 = -1

    coefficients = [a3, a2, a1, a0]
    return np.roots(coefficients)


def edges_rho(sigma, rho):
    alpha = 1/rho
    R_noise = sigma

    a0 = -12 * R_noise + (4 * R_noise) / alpha + 12 * alpha * R_noise - 4 * alpha**2 * R_noise - 20 * R_noise**2 + R_noise**2 / alpha - 8 * alpha * R_noise**2 - 4 * R_noise**3
    a1 = -(10 * R_noise) / np.sqrt(alpha) + 2 * np.sqrt(alpha) * R_noise + 8 * alpha**(3/2) * R_noise - (2 * R_noise**2) / np.sqrt(alpha) + 8 * np.sqrt(alpha) * R_noise**2
    a2 = 1 - 2 * alpha + alpha**2 + 8 * R_noise - 2 * alpha * R_noise + R_noise**2
    a3 = -2 * np.sqrt(alpha) - 2 * alpha**(3/2) - 2 * np.sqrt(alpha) * R_noise
    a4 = alpha

    coefficients = [a4, a3, a2, a1, a0]

    roots_all = np.roots(coefficients)
    real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-6])

    return np.sort(real_roots)



def spectral_density(x, Delta, rho):        
    return np.max(np.imag(stieltjes_root(x-1e-8j, Delta, rho))) / np.pi


def integral_mu_cubed(hat_q, rho):
    
    Delta = 1/np.abs(hat_q) # Senza sqrt() qui vuole il quadrato !
    edges_list = edges_rho(Delta, rho)

    if len(edges_list) == 4:
        return quad(lambda x: spectral_density(x, Delta, rho)**3, edges_list[0], edges_list[1], epsabs=1e-4, epsrel=1e-4)[0] + quad(lambda x: spectral_density(x, Delta, rho)**3, edges_list[2], edges_list[3], epsabs=1e-4, epsrel=1e-4)[0]
    else:
        return quad(lambda x: spectral_density(x, Delta, rho)**3, edges_list[0], edges_list[1], epsabs=1e-4, epsrel=1e-4)[0]


def solve_prior_fixed_hatq(hat_q, Q0, rho):
    
    val = integral_mu_cubed(hat_q, rho)
    q_raw = Q0 - (1.0/hat_q) + (4.0*np.pi**2/(3.0*hat_q**2))*val
    return q_raw


def bvn_cdf_pdf(k1, k2, rho):

    cov  = [[1.0, rho], [rho, 1.0]]
    Phi2 = multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([k1, k2])
    denom = 2.0 * np.pi * np.sqrt(1.0 - rho * rho)
    expo  = -(k1 * k1 - 2.0 * rho * k1 * k2 + k2 * k2) / (2.0 * (1.0 - rho * rho))
    phi2  = np.exp(expo) / denom
    return Phi2, phi2


def sum_G_squared_hard(eta, y, q, Q):

    omega11 = np.sqrt(2*q) * eta[0, 0]
    omega12 = np.sqrt(  q) * eta[0, 1]
    omega22 = np.sqrt(2*q) * eta[1, 1]
    V = Q - q
    r = np.sqrt(3*V)

    s11 = 1.0 if y[0, 0] == 1.0 else -1.0
    s22 = 1.0 if y[1, 1] == 1.0 else -1.0

    kappa1   = s11 * (omega11 - omega12) / r
    kappa2   = s22 * (omega22 - omega12) / r
    rho12  = s11 * s22 /3 

    Phi2, pdf = bvn_cdf_pdf(kappa1, kappa2, rho12)
    invphi = 1.0 / Phi2                                 


    c11 =  s11 / r
    c22 =  s22 / r
    c12 = -(s11  + s22 ) / r
 
    
    G11  = c11 * pdf * invphi
    G22  = c22 * pdf * invphi
    G12  =  c12 * pdf * invphi

    return 2*G11*G11 +2* G22*G22 + G12*G12





def mc_hatq_hard(alpha, q, Q0, nsamples=25000):
    acc = 0.0
    for _ in range(nsamples):
        eta  = rng.standard_normal((2, 2))                 
        xi  = rng.standard_normal((2, 2))

        h  = np.empty((2, 2))
        h[0, 0] = np.sqrt(2*q) * eta[0, 0] + np.sqrt(2*(Q0 - q)) * xi[0, 0]
        h[0, 1] = np.sqrt(  q) * eta[0, 1] + np.sqrt(  (Q0 - q)) * xi[0, 1]
        h[1, 1] = np.sqrt(2*q) * eta[1, 1] + np.sqrt(2*(Q0 - q)) * xi[1, 1]
        h[1, 0] = h[0, 1]

        y = np.zeros((2, 2))
        y[0, 0] = 1.0 if h[0, 0] > h[0, 1] else 0.0
        y[0, 1] = 1.0 - y[0, 0]
        y[1, 1] = 1.0 if h[1, 1] > h[1, 0] else 0.0
        y[1, 0] = 1.0 - y[1, 1]

        acc += sum_G_squared_hard(eta, y, q, Q0)

    mean_G2 = acc / nsamples
    return (2.0 * alpha ) * mean_G2



                
def solve_state_eq_hard_iters(alpha, rho,
                              q_init   = 0.30,
                              max_iter = 1000,
                              damping  = 0.5,
                              nsamp_out= 25000,
                              tol      = 1e-5):
    Q0 = 1 + rho
    q  = rho
    history = [q]
    for t in range(max_iter+1):
        hatq  = mc_hatq_hard(alpha, q, Q0, nsamp_out)
        q_new = solve_prior_fixed_hatq(hatq, Q0, rho)
        q_next= damping*q_new + (1-damping)*q
        history.append(q_next)

        pd.DataFrame({"q": history}).to_csv(f"SE_hardmax_{rho}/alpha_{alpha}.csv", index=False)

        if t >= 150 and abs(q_next - q) < tol:
            break
        q = q_next
    return np.array(history)

alpha_idx = int(argv[1])
rho       = float(argv[2])

alpha_grid= np.logspace(-3, np.log10(0.02), 128)
# alpha_grid= np.linspace(0.02, 10, 100)
alpha = alpha_grid[alpha_idx]

Q0    = 1 + rho

solve_state_eq_hard_iters(alpha, rho)
