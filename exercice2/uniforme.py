import math
import matplotlib.pyplot as plt
import numpy as np

def gen_uniforme(N):
    """Génère U ~ Uniforme(0,1)"""
    return np.random.random(N)

def sim_tcl_uniforme(n, N):
    """
    S_n = somme de n variables uniformes.
    Z_n = (S_n - n*mu) / (sigma * sqrt(n))
    Pour U[0,1]: mu = 0.5, sigma^2 = 1/12
    """
    mu = 0.5
    sigma = np.sqrt(1/12)

    samples = []
    for _ in range(N):
        sum_x = np.sum(gen_uniforme(n))
        z = (sum_x - n * mu) / (sigma * np.sqrt(n))
        samples.append(z)
    return np.array(samples)

def gaussian_density(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)

def analyze_and_plot_tcl(data, bins=30):
    """
    theorical esperence for expo = 1/lambda
    theorical var for expo = 1/lambda^2
    :param data:
    :param lmbda:
    :param bins:
    :return:
    """
    mean_emp = np.mean(data)
    var_emp = np.var(data)

    print(f"Moyenne empirique : {mean_emp} espérance, {0}")
    print(f"Variance empirique : {var_emp} Variance, {1}")

    plt.figure(figsize=(8, 5))
    count, bins, ignored = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.linspace(min(bins), max(bins), 200)
    y_vals = [gaussian_density(x) for x in x_vals]
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Densité Théorique')

    plt.title(f"Simulation : TCL (Uniforme)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    N = 10000
    n = 100
    samples = sim_tcl_uniforme(n, N)
    analyze_and_plot_tcl(samples)
