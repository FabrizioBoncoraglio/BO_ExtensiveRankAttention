import numpy as np
import pandas as pd
from scipy.integrate import quad
from tqdm import tqdm
import sys
from numpy.random import default_rng
from scipy.integrate import quad
from scipy.stats import multivariate_normal


rng = default_rng()



def solve_poly(z, sigma, kappa):
    alpha = 1 / kappa
    R_noise = sigma
    a3 = np.sqrt(alpha) * R_noise
    a2 = -(np.sqrt(alpha) * z + R_noise)
    a1 = (z + np.sqrt(alpha) - alpha**(-1 / 2))
    a0 = -1

    coefficients = [a3, a2, a1, a0]

    # Find the roots of the polynomial
    return np.roots(coefficients)


def edges_rho(sigma, kappa):
    alpha = 1/kappa
    R_noise = sigma

    a0 = -12 * R_noise + (4 * R_noise) / alpha + 12 * alpha * R_noise - 4 * alpha**2 * R_noise - 20 * R_noise**2 + R_noise**2 / alpha - 8 * alpha * R_noise**2 - 4 * R_noise**3
    a1 = -(10 * R_noise) / np.sqrt(alpha) + 2 * np.sqrt(alpha) * R_noise + 8 * alpha**(3/2) * R_noise - (2 * R_noise**2) / np.sqrt(alpha) + 8 * np.sqrt(alpha) * R_noise**2
    a2 = 1 - 2 * alpha + alpha**2 + 8 * R_noise - 2 * alpha * R_noise + R_noise**2
    a3 = -2 * np.sqrt(alpha) - 2 * alpha**(3/2) - 2 * np.sqrt(alpha) * R_noise
    a4 = alpha

    # Coefficients of the polynomial
    coefficients = [a4, a3, a2, a1, a0]

    roots_all = np.roots(coefficients)
    real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-6])

    return np.sort(real_roots)


def rho(x, sigma, kappa):        
    return np.max(np.imag(solve_poly(x-1e-8j, sigma, kappa))) / np.pi

def integral_rho(Delta, kappa):
        
    def rho(x):        
        return np.max(np.imag(solve_poly(x-1e-8j, Delta, kappa))) / np.pi

    
    edges_list = edges_rho(Delta, kappa)

    if len(edges_list) == 4:
        return quad(lambda x: rho(x)**3, edges_list[0], edges_list[1])[0] + quad(lambda x: rho(x)**3, edges_list[2], edges_list[3])[0]
    else:
        return quad(lambda x: rho(x)**3, edges_list[0], edges_list[1])[0]



def f_RIE(R, Delta, kappa):
    Delta = Delta +1e-6
    def denoiser(x):        
        choose_root = np.argmax(np.imag(solve_poly(x-1e-8j, Delta, kappa))) 
        return np.real(solve_poly(x-1e-8j, Delta, kappa))[choose_root]
    
    eigval, eigvec = np.linalg.eig(R)
    eigval_denoised = np.array([e - 2*Delta*denoiser(e) for e in eigval])
    return eigvec @ np.diag(eigval_denoised) @ eigvec.T


def F_RIE(Delta, kappa):
    return Delta - 4*np.pi**2/3 * Delta**2 * integral_rho(Delta, kappa) #why Delta e non -Delta +1 +rho (io ho Q_0 - 1/\hat{q} + int mu^3)


def hardmax(h):
    flat = h.reshape(-1, h.shape[-1])
    idx  = np.argmax(flat, axis=1)
    one_hot = np.zeros_like(flat)
    one_hot[np.arange(flat.shape[0]), idx] = 1
    return one_hot.reshape(h.shape)


L=2
def data_generation(D, rho, alpha, L=L):
    R = int(rho * D)
    N = int(alpha * D**2)

    X_mu = np.random.normal(0,1, (N,D,L))
    X_test = np.random.normal(0,1, (N,D,L))
    
    X = np.einsum("mia,mjb->mijab", X_mu,X_mu) - np.einsum("m,ij,ab->mijab", np.ones(N),np.eye(D),np.eye(L))
    X_test = np.einsum("mia,mjb->mijab", X_test,X_test) - np.einsum("m,ij,ab->mijab", np.ones(N),np.eye(D),np.eye(L))
    
    X = (np.einsum("mijab->mjiab", X)+ X) / np.sqrt(2)
    X_test = (np.einsum("mijab->mjiab", X_test)+ X_test) / np.sqrt(2)

    for a in range(L):
        X[:,:,:,a , a] /= np.sqrt(2) 
        X_test[:,:,:,a , a] /= np.sqrt(2)

    W = np.random.normal(0,1, (D, R))
    S = W @ W.T / np.sqrt(R)

    h = np.einsum('mijab,ij->mab', X, S) / D
    h_test = np.einsum('mijab,ij->mab', X_test, S) /D

    h_reshaped = h.reshape(-1, h.shape[-1])  
    y      = hardmax(h)
    y_test = hardmax(h_test)

    # return X, y, X_test, y_test, S
    return X, y, S

from scipy.stats import multivariate_normal

def bvn_pdf_cdf(k1, k2, rho):
    Phi2 = multivariate_normal(
               mean=[0.0, 0.0],
               cov=[[1.0, rho], [rho, 1.0]]
           ).cdf([k1, k2])
    denom = 2.0 * np.pi * np.sqrt(1.0 - rho * rho)
    expo  = -(k1*k1 - 2*rho*k1*k2 + k2*k2) / (2.0*(1.0 - rho*rho))
    if expo < -700:  
        phi2 = 0.0
    else:
        phi2 = np.exp(expo) / denom

    return Phi2, phi2


def gOut(Y, omega, V):
    N = Y.shape[0]
    G = np.zeros_like(omega)

    r = np.sqrt(2*V)

    for i in range(N):
        s11 = +1.0 if Y[i,0,0] == 1.0 else -1.0
        s22 = +1.0 if Y[i,1,1] == 1.0 else -1.0

        k1 = s11 * (omega[i,0,0] - omega[i,0,1]) / r
        k2 = s22 * (omega[i,1,1] - omega[i,0,1]) / r
        rho = s11 * s22 /2 

        Phi2, phi2 = bvn_pdf_cdf(k1, k2, rho)
        ratio = phi2 / max(Phi2, 1e-300)      

        g11 =  s11 / r * ratio
        g22 =  s22 / r * ratio
        g12 = -(s11 / r + s22 / r) * ratio

        G[i,0,0] = g11
        G[i,1,1] = g22
        G[i,0,1] = g12
        G[i,1,0] = 0 

    return G


def AMP(X, y, S, beta, iterations = 500, damping = 0.4, tol = 1e-5, verbose = False):

    N, D, _, L, _ = X.shape
    mask = np.array([[1,1],[0,1]])

    r = int(D*beta)
    alpha = N/D**2
        
    uX = X / np.sqrt(D) # X_mu has O(1) components and y_mu is O(1). We normalise X to have simpler equations later
        
    # hatS has O(1) SPECTRUM
    W = np.random.normal(0,1, (D, r))
    hatS = W @ W.T / np.sqrt(r) / np.sqrt(D) 

    if verbose:
        # print(f"==> Squared norm of iterate is {np.linalg.norm(hatS)**2 / D}, which is compatible with the theory: {1 + r/D}")
        print(f"==> Squared norm distance with true S is {np.linalg.norm(S - hatS*np.sqrt(D))**2 / D**2}")
        
    hatC    = 10.
    omega   = np.ones((N,L,L))*10.
    V       = 10.

    error = np.inf
    for t in range(iterations):
        newV = 2*hatC
        newOmega = np.einsum("nijab,ij->nab", uX, hatS) - np.einsum("nab,ab->nab", gOut(y, omega, V), mask) * newV  
        
        V = newV * (1-damping) + V * damping
        omega = newOmega * (1-damping) + omega * damping
        
        A_normalised = np.einsum("nab,ab->", gOut(y, omega, V)**2, mask) * alpha / N * 2

        R = hatS + 1 / (A_normalised * D)  * np.einsum("nijab,nab,ab->ij", uX, gOut(y, omega, V),mask)
        
        # Factor 2
        noise_A = 1 / A_normalised / 2 
        newhatS = f_RIE(R, noise_A, r/D)
        hatC = F_RIE(noise_A, r/D)  #* 2

        
        error = np.linalg.norm(hatS - newhatS)**2 / D
        error_eval = np.linalg.norm(hatS*np.sqrt(D) - S)**2 / D**2
        hatS = newhatS

        if verbose:
            print(f"--> Squared norm distance of iteration step is {error}")
            print(f"--> Squared norm distance of true S is {error_eval}")
        
        if error < tol:
            break

    return hatS, error_eval


def AMP_experiment(D, alpha, rho, samples=2, iterations=100, damping=0.5, tol=1e-5, verbose=False, L=2):
    errors = []
    for i in tqdm(range(samples)):
        # Remove noise from the function call
        X, y, S = data_generation(D, rho, alpha, L=L)
       
        _, error = AMP(X, y, S, rho, iterations=iterations, damping=damping, tol=tol, verbose=False)
        errors.append(error)

        if verbose:
            print(f"Error is {errors[i]}")

        pd.DataFrame({"error": errors}).to_csv(f"AMP_hardmax/alpha_{alpha}_rho_{rho}.csv", index=False)
    return errors


if __name__ == "__main__":
    alpha_idx  = int(sys.argv[1])
    rho_val    = float(sys.argv[2])
    alpha_grid = np.logspace(np.log10(0.03), np.log10(8), 32)
    # alpha_grid= np.linspace(0.02, 10, 100)
    
    alpha     = alpha_grid[alpha_idx]
    D         = 100
    samples   = 16
    iterations= 500
    damping   = 0.5
    tol       = 1e-5

    AMP_experiment(D, alpha, rho_val, samples=samples, iterations=iterations, damping=damping, tol=tol, verbose=False)