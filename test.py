import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Configuration générale
np.random.seed(42)  # Pour la reproductibilité
N = 1000            # Taille de l'échantillon (>= 500 demandé)

# ==============================================================================
# 1. OUTILS GÉNÉRAUX ET EXERCICE 1 : MÉTHODE D'INVERSION
# ==============================================================================

def generer_uniforme(n_samples):
    """Génère U ~ Uniforme(0,1)"""
    return np.random.random(n_samples)

# 1.1 Loi Binomiale B(n, p) par inversion de la fonction de répartition
def inversion_binomiale(n_trials, p, size):
    # Note: Pour une variable discrète, l'inversion revient à trouver k tel que F(k-1) < U <= F(k)
    samples = []
    for _ in range(size):
        u = np.random.random()
        cdf = 0.0
        k = 0
        while k <= n_trials:
            # pmf = (n choose k) * p^k * (1-p)^(n-k)
            pmf = math.comb(n_trials, k) * (p**k) * ((1-p)**(n_trials-k))
            cdf += pmf
            if u <= cdf:
                samples.append(k)
                break
            k += 1
    return np.array(samples)

# 1.2 Loi de Poisson(lam) par inversion
def inversion_poisson(lam, size):
    samples = []
    for _ in range(size):
        u = np.random.random()
        k = 0
        pmf = math.exp(-lam) # P(X=0)
        cdf = pmf
        while u > cdf:
            k += 1
            pmf *= lam / k # Récurrence P(X=k) = P(X=k-1) * lambda / k
            cdf += pmf
        samples.append(k)
    return np.array(samples)

# 1.3 Loi Exponentielle(theta) par inversion
# F(x) = 1 - exp(-theta * x) => x = -ln(1-U)/theta. Comme 1-U ~ U, on utilise -ln(U)/theta
def inversion_exponentielle(theta, size):
    u = generer_uniforme(size)
    return -np.log(u) / theta

# 1.4 Loi continue de densité f(x) = 2x sur [0,1]
# F(x) = x^2 => x = sqrt(U)
def inversion_densite_2x(size):
    u = generer_uniforme(size)
    return np.sqrt(u)

# ==============================================================================
# 2. EXERCICE 2 : THÉORÈME CENTRAL LIMITE (TCL)
# ==============================================================================

def simulation_tcl_uniforme(n_termes, size):
    """
    S_n = somme de n variables uniformes.
    Z_n = (S_n - n*mu) / (sigma * sqrt(n))
    Pour U[0,1]: mu = 0.5, sigma^2 = 1/12
    """
    mu = 0.5
    sigma = np.sqrt(1/12)

    samples = []
    for _ in range(size):
        # Somme de n variables uniformes
        sum_x = np.sum(generer_uniforme(n_termes))
        z = (sum_x - n_termes * mu) / (sigma * np.sqrt(n_termes))
        samples.append(z)
    return np.array(samples)

def simulation_tcl_exponentielle(n_termes, size, theta):
    """
    Même chose mais en partant d'une loi exponentielle.
    Pour Exp(theta): mu = 1/theta, sigma = 1/theta
    """
    mu = 1/theta
    sigma = 1/theta

    samples = []
    for _ in range(size):
        # Somme de n variables exponentielles (générées par notre méthode d'inversion)
        sum_x = np.sum(inversion_exponentielle(theta, n_termes))
        z = (sum_x - n_termes * mu) / (sigma * np.sqrt(n_termes))
        samples.append(z)
    return np.array(samples)

# ==============================================================================
# 3. EXERCICE 3 : BOX-MULLER
# ==============================================================================

def box_muller_standard(size):
    # On a besoin de paires, donc on génère size si pair, size+1 si impair
    count = size if size % 2 == 0 else size + 1
    u = generer_uniforme(count // 2)
    v = generer_uniforme(count // 2)

    r = np.sqrt(-2 * np.log(u))
    x = r * np.cos(2 * np.pi * v)
    y = r * np.sin(2 * np.pi * v)

    return np.concatenate([x, y])[:size]

def box_muller_general(mu, sigma, size):
    standard = box_muller_standard(size)
    return standard * sigma + mu

# ==============================================================================
# FONCTIONS D'AFFICHAGE ET D'ANALYSE
# ==============================================================================

def analyze_and_plot(data, name, theoretical_mean=None, theoretical_var=None, bins=30, density_func=None):
    print(f"--- Analyse : {name} ---")
    mean_emp = np.mean(data)
    var_emp = np.var(data)
    print(f"Moyenne empirique : {mean_emp:.4f} (Théorique : {theoretical_mean if theoretical_mean else 'N/A'})")
    print(f"Variance empirique: {var_emp:.4f} (Théorique : {theoretical_var if theoretical_var else 'N/A'})")

    plt.figure(figsize=(8, 5))
    count, bins, ignored = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogramme Simulation')

    # Ajout de la densité théorique si fournie
    if density_func:
        if name == "Loi Binomiale" or name == "Loi Poisson":
            # Pour discret, on affiche des points ou barres rouges
            x_vals = np.arange(min(data), max(data)+1)
            y_vals = [density_func(k) for k in x_vals]
            plt.plot(x_vals, y_vals, 'ro', label='Théorie')
        else:
            x_vals = np.linspace(min(bins), max(bins), 200)
            y_vals = [density_func(x) for x in x_vals]
            plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Densité Théorique')

    plt.title(f"Simulation : {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plot_{name.replace(' ', '_')}.png") # Sauvegarde pour le rapport
    plt.show()
    print("\n")

# ==============================================================================
# MAIN - EXÉCUTION DU TP
# ==============================================================================

if __name__ == "__main__":

    # --- Exercice 1 : Inversion ---

    # 1. Binomiale (n=10, p=0.3)
    n_bin, p_bin = 10, 0.3
    theo_mean_bin = n_bin * p_bin
    theo_var_bin = n_bin * p_bin * (1 - p_bin)
    data_bin = inversion_binomiale(n_bin, p_bin, N)
    analyze_and_plot(data_bin, "Loi Binomiale", theo_mean_bin, theo_var_bin, bins=range(0, n_bin+2),
                     density_func=lambda k: math.comb(n_bin, int(k)) * (p_bin**k) * ((1-p_bin)**(n_bin-k)))

    # 2. Poisson (lambda=2)
    lam_pois = 2
    data_pois = inversion_poisson(lam_pois, N)
    analyze_and_plot(data_pois, "Loi Poisson", lam_pois, lam_pois, bins=range(0, max(data_pois)+2),
                     density_func=lambda k: (np.exp(-lam_pois) * lam_pois**k) / math.factorial(int(k)))

    # 3. Exponentielle (theta=2)
    theta_exp = 2
    data_exp = inversion_exponentielle(theta_exp, N)
    analyze_and_plot(data_exp, "Loi Exponentielle", 1/theta_exp, 1/(theta_exp**2),
                     density_func=lambda x: theta_exp * np.exp(-theta_exp * x) if x >= 0 else 0)

    # 4. Densité f(x)=2x
    # Moyenne = Integrale(x * 2x) = [2/3 x^3] de 0 à 1 = 2/3
    # E(X^2) = Integrale(x^2 * 2x) = [2/4 x^4] = 1/2
    # Var = 1/2 - (2/3)^2 = 1/2 - 4/9 = 9/18 - 8/18 = 1/18
    data_2x = inversion_densite_2x(N)
    analyze_and_plot(data_2x, "Densite f(x)=2x", 2/3, 1/18,
                     density_func=lambda x: 2*x if 0 <= x <= 1 else 0)

    # --- Exercice 2 : TCL ---

    # 1. TCL Uniforme
    n_terms = 100 # n >= 100
    data_tcl = simulation_tcl_uniforme(n_terms, N)
    gaussian_pdf = lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    analyze_and_plot(data_tcl, "TCL (Source Uniforme)", 0, 1, density_func=gaussian_pdf)

    # 2. TCL Exponentielle
    data_tcl_exp = simulation_tcl_exponentielle(n_terms, N, theta_exp)
    analyze_and_plot(data_tcl_exp, "TCL (Source Exponentielle)", 0, 1, density_func=gaussian_pdf)

    # --- Exercice 3 : Box-Muller ---

    # 1. Standard N(0,1)
    data_bm = box_muller_standard(N)
    analyze_and_plot(data_bm, "Box-Muller N(0,1)", 0, 1, density_func=gaussian_pdf)

    # 2. Quelconque N(2, 0.5) (Var = 0.5 donc sigma = sqrt(0.5))
    mu_target, sigma_sq_target = 2, 0.5
    sigma_target = np.sqrt(sigma_sq_target)
    data_bm_custom = box_muller_general(mu_target, sigma_target, N)

    pdf_custom = lambda x: (1/(sigma_target*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu_target)/sigma_target)**2)
    analyze_and_plot(data_bm_custom, f"Box-Muller N({mu_target}, {sigma_sq_target})", mu_target, sigma_sq_target, density_func=pdf_custom)