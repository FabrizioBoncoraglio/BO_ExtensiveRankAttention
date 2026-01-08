import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

def stieltjes_root(z, sigma, rho):
    alpha = 1 / rho
    R_noise = sigma
    a3 = np.sqrt(alpha) * R_noise
    a2 = -(np.sqrt(alpha) * z + R_noise) #
    a1 = (z + np.sqrt(alpha) - alpha**(-1 / 2))
    a0 = -1

    # Coefficients of the polynomial
    coefficients = [a3, a2, a1, a0]

    # Find the roots of the polynomial
    return np.roots(coefficients)

def edges_rho(sigma, rho):
    alpha = 1/rho
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


def spectral_density(x, Delta, rho):        
    return np.max(np.imag(stieltjes_root(x-1e-8j, Delta, rho))) / np.pi


def integral_mu_cubed(hat_q, rho):
    
    Delta = 1/np.abs(hat_q) 
    edges_list = edges_rho(Delta, rho)

    if len(edges_list) == 4:
        return quad(lambda x: spectral_density(x, Delta, rho)**3, edges_list[0], edges_list[1], epsabs=1e-4, epsrel=1e-4)[0] + quad(lambda x: spectral_density(x, Delta, rho)**3, edges_list[2], edges_list[3], epsabs=1e-4, epsrel=1e-4)[0]
    else:
        return quad(lambda x: spectral_density(x, Delta, rho)**3, edges_list[0], edges_list[1], epsabs=1e-4, epsrel=1e-4)[0]



def solve_prior_fixed_hatq(hat_q, Q0, rho):
    
    val = integral_mu_cubed(hat_q, rho)
    q_raw = Q0 - (1.0/hat_q) + (4.0*np.pi**2/(3.0*hat_q**2))*val
    return q_raw


def output_eq_exact(alpha, q_cur, Q0, L) :

    Q_minus_q = (Q0 - q_cur)
    term2 = (L**2 + L - 2)* alpha  / (Q_minus_q)
    return term2

def solve_state_equations(alpha, rho,
                          L,
                          Q0 = None,
                          q_init = 0.3,
                          max_iter = 40,
                          nsamp_out = 2000,
                          damping = 0.3,
                          tol=1e-5) :

    if Q0 is None:
        Q0 = 1.0 + rho

    q_cur = q_init

    for it in range(max_iter):
        hat_q_1 = output_eq_exact(alpha, q_cur, Q0, L)
        hat_q = 0.7 * hat_q_1+ 0.3 * (hat_q if it > 0 else hat_q_1)

        q_new = solve_prior_fixed_hatq(hat_q, Q0, rho)

        q_next = damping*q_new + (1.0-damping)*q_cur
        q_next = q_next
        delta_q = abs(q_next - q_cur)
        delta_hat_q = abs(hat_q - output_eq_exact(alpha, q_next, Q0, L))

        print(f"Alpha={alpha}, Iter={it}, q_cur={q_cur:.5f}, hat_q={hat_q:.5g}, q_new={q_new:.5f}, q_next={q_next:.5f}, ",
              f"delta_q={delta_q:.5g}, delta_hat_q={delta_hat_q:.5g}")

        if delta_q < tol: 
            print("Convergence reached.")
            break

        q_cur = q_next

    return q_cur


def run_q_and_mmse_vs_alpha(rho_list,
                      alpha_list,
                      L,
                      q_init,
                      max_iter,
                      damping):
    fig, ax_mmse = plt.subplots(figsize=(6, 4))
    all_mmse_vals = []
    q_values = []
    all_q = []

    for rho in rho_list:
        Q0 = 1.0 + rho
        mmse_vals = []
        q_values = []
        for alpha in alpha_list:
            q_sol = solve_state_equations(alpha, rho, L,
                                           q_init=q_init,
                                           max_iter=max_iter,
                                           damping=damping)
            mmse_vals.append(Q0 - q_sol)
            q_values.append(q_sol)

        ax_mmse.plot(alpha_list, mmse_vals, '-', label=rf"$\rho={rho}$", linewidth=1)
        all_mmse_vals.append(mmse_vals)
        all_q.append(q_values)

    ax_mmse.set_xlabel(r"$\alpha$")
    ax_mmse.set_ylabel("MMSE = Q0 - q")
    ax_mmse.set_title(rf"Linear channel, L={L}: MMSE vs $\alpha$")
    ax_mmse.legend()
    # plt.savefig('mmse_vs_alpha_linear.pdf')


    plt.tight_layout()
    plt.show()
    return all_mmse_vals, all_q

rho_list = [0.5]
L = 2

alpha_min = 2e-5
alpha_max = (rho_list[0] - rho_list[0]**2/2)/(L**2+L-2)*2

alpha_fine = np.linspace(alpha_min*rho_list[0], rho_list[0]/(L**2+L-2)*2, 256) 


mmse_vals, all_q = run_q_and_mmse_vs_alpha(rho_list, alpha_fine, L=L, q_init=0.3, max_iter=5000, damping=0.5) 

# Save the data as you prefer!