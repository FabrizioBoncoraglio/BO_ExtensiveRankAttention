import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from mpmath import mp
from scipy.integrate import quad


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
    Delta = Delta + 1e-6
    def denoiser(x):        
        choose_root = np.argmax(np.imag(solve_poly(x-1e-8j, Delta, kappa))) 
        return np.real(solve_poly(x-1e-8j, Delta, kappa))[choose_root]
    
    eigval, eigvec = np.linalg.eig(R)
    eigval_denoised = np.array([e - 2*Delta*denoiser(e) for e in eigval])
    return eigvec @ np.diag(eigval_denoised) @ eigvec.T


def F_RIE(Delta, kappa):
    return Delta - 4*np.pi**2/3 * Delta**2 * integral_rho(Delta, kappa) #why Delta e non -Delta +1 +rho (io ho Q_0 - 1/\hat{q} + int mu^3)



def softmax(x, temp=1):
    max_x = np.max(x, axis=-1, keepdims=True)
    x = x - max_x
    P = np.exp(temp * x)
    return P / np.sum(P, axis=-1, keepdims=True)

#change
def phi_l2(y, temp=1.0):
    phi = np.zeros_like(y, dtype=np.float64)
    phi[..., 0, 0] = float(mp.log(float(y[..., 0, 0])) - mp.log(float(y[..., 0, 2])))/ temp
    phi[..., 0, 1] = float(mp.log(float(y[..., 0, 1])) - mp.log(float(y[..., 0, 2])))/ temp
    phi[..., 0, 2] = 0.0
    phi[..., 1, 0] = float(mp.log(float(y[..., 1, 0])) - mp.log(float(y[..., 1, 2])))/ temp
    phi[..., 1, 1] = float(mp.log(float(y[..., 1, 1])) - mp.log(float(y[..., 1, 2])))/ temp
    phi[..., 1, 2] = 0.0
    phi[..., 2, 0] = float(mp.log(float(y[..., 2, 0])) - mp.log(float(y[..., 2, 2])))/ temp
    phi[..., 2, 1] = float(mp.log(float(y[..., 2, 1])) - mp.log(float(y[..., 2, 2])))/ temp
    phi[..., 2, 2] = 0.0
    return phi

L=3
def data_generation(D, rho, alpha, L=L):
    R = int(rho * D)
    N = int(alpha * D**2)

    X_mu = np.random.normal(0,1, (N,D,L))
    X_test = np.random.normal(0,1, (N,D,L))


    #construct the x
    X = np.einsum("mia,mjb->mijab", X_mu,X_mu) - np.einsum("m,ij,ab->mijab", np.ones(N),np.eye(D),np.eye(L))
    X_test = np.einsum("mia,mjb->mijab", X_test,X_test) - np.einsum("m,ij,ab->mijab", np.ones(N),np.eye(D),np.eye(L))

    W = np.random.normal(0,1, (D, R))
    S = W @ W.T / np.sqrt(R)
    

    h = np.einsum('mijab,ij->mab', X, S) / D
    h_test = np.einsum('mijab,ij->mab', X_test, S) /D
    
    X = (np.einsum("mijab->mjiab", X)+ X) / np.sqrt(2)
    X_test = (np.einsum("mijab->mjiab", X_test)+ X_test) / np.sqrt(2)

    for a in range(L):
        X[:,:,:,a , a] /= np.sqrt(2) # put sqrt(2)
        X_test[:,:,:,a , a] /= np.sqrt(2)


    h_reshaped = h.reshape(-1, h.shape[-1])  
    y = softmax(h_reshaped).reshape(h.shape)  
    y_test = softmax(h_test.reshape(-1, h_test.shape[-1])).reshape(h_test.shape)    


    return X, y, X_test, y_test, S

_tau = np.ones((3, 3), dtype=float)
_tau[np.triu_indices(3, k=1)] = np.sqrt(2.0)

def gOut(Y, W, V):
    eps   = 1e-40
    denom = Y[:, :, 2:3] + eps
    phi   = np.log((Y + eps) / denom)           
    tau       = np.ones((3, 3))
    tau[np.triu_indices(3, k=1)] = np.sqrt(2.0)
    tau2      = tau**2
    iu        = np.triu_indices(3)            
    tau_flat  = tau[iu]                     
    tau2_flat = tau2[iu]

    tau2_sum_per_c = np.array([tau2_flat[0] + tau2_flat[1] + tau2_flat[2],
                               tau2_flat[3] + tau2_flat[4],
                               tau2_flat[5]])

    phi_Tc = phi[:, 2, :]                      

    S1 = (phi_Tc * tau2_sum_per_c).sum(axis=1)                # (N,)
    S2 = (W[:, iu[0], iu[1]] * tau_flat).sum(axis=1)           # (N,)
    d_le_1_mask = iu[1] <= 1                                   # keep d = 0,1
    S3 = (phi[:, iu[0][d_le_1_mask], iu[1][d_le_1_mask]]
          * tau2_flat[d_le_1_mask]).sum(axis=1)                # (N,)

    common = (S1 - S2 + S3) / 9.0                              # divide by T²
    common = common[:, None, None]                             # (N,1,1) for broadcasting

    delta_b_lt_T = np.array([[1., 1., 0.],
                             [0., 1., 0.],
                             [0., 0., 0.]])

    tau_broadcast = tau[None, :, :]                            # (1,3,3)
    term2 = (tau_broadcast * phi_Tc[:, :, None]        # τ_ab φ_{Ta}
             - W
             + tau_broadcast * phi * delta_b_lt_T)     # +1_{b<T} τ_ab φ_{ab}

    G = -tau_broadcast * common / V + term2 / V

    G[:, 1, 0] = 0.0
    G[:, 2, 0] = 0.0
    G[:, 2, 1] = 0.0
    return G


def AMP(X, y, X_test, y_test, S, beta, iterations = 100, damping = 0.4, tol = 1e-5, verbose = False):

    N, D, _, L, _ = X.shape
    mask = np.array([[1,1,1],[0,1,1],[0,0,1]])

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

        h_test = np.einsum('mijab,ij->mab', X_test, hatS) /np.sqrt(D)

        error = np.linalg.norm(hatS - newhatS)**2 / D
        error_eval = np.linalg.norm(hatS*np.sqrt(D) - S)**2 / D**2
        y_test_hat = softmax(h_test.reshape(-1, h_test.shape[-1])).reshape(h_test.shape)  
        e_test = np.linalg.norm(y_test - y_test_hat)**2 / N
        hatS = newhatS

        if verbose:
            print(f"--> Squared norm distance of iteration step is {error}")
            print(f"--> Squared norm distance of true S is {error_eval}")
        
        if error < tol:
            break

    return hatS, error_eval, e_test

L=3

def AMP_experiment(D, alpha, rho, samples=2, iterations=100, damping=0.5, tol=1e-5, verbose=False):
    errors = np.zeros(samples)
    e_test = np.zeros(samples)
    for i in tqdm(range(samples)):
        # Remove noise from the function call
        X, y, X_test, y_test, S = data_generation(D, rho, alpha, L=L)
        _, errors[i], e_test[i] = AMP(X, y, X_test, y_test, S, rho, iterations=iterations, damping=damping, tol=tol, verbose=False)

        if verbose:
            print(f"Error is {errors[i]}")
    return errors, e_test


D = 150
rho = 0.5
alpha = 0.18
noise = 0.0
samples = 20
alpha_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12]


for alpha in alpha_list:

    errors, e_test = AMP_experiment(D, alpha, rho, samples = samples, iterations = 100, damping = 0.5, tol = 1e-5, verbose=True)

    print(f"Mean error is {errors.mean()} and std is {errors.std()}")
    print(f"{alpha},{errors.mean()},{errors.std()}")
    print("\n")
    print(f"Mean test error is {e_test.mean()} and std is {e_test.std()}")
    print(f"{alpha},{e_test.mean()},{e_test.std()}")