import matplotlib.pyplot as plt
import numpy as np
import math

def gen_uniforme(n_samples):
    return np.random.random(n_samples)

def sim_box_muller_standard(size):
    count = size if size % 2 == 0 else size + 1
    u = gen_uniforme(count // 2)
    v = gen_uniforme(count // 2)

    r = np.sqrt(-2 * np.log(u))
    x = r * np.cos(2 * np.pi * v)
    y = r * np.sin(2 * np.pi * v)

    return np.concatenate([x, y])[:size] # Concact cause size is too high

def sim_box_muller_general(mu, sigma, N):
    standard = sim_box_muller_standard(N)
    return standard * sigma + mu

def custom_density(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2)

def analyze_and_plot_poisson(data, mu, sigma):
    """
    :param data:
    :param lmbda:
    :param bins:
    :return:
    """
    mean_emp = np.mean(data)
    var_emp = np.var(data)
    print(f"Moyenne empirique : {mean_emp} espérance, {mu}")
    print(f"Variance empirique : {var_emp} Variance, {sigma**2}")

    plt.figure(figsize=(8, 5))

    count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.linspace(min(bins), max(bins), 200)
    y_vals = [custom_density(x,mu,sigma) for x in x_vals]
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Densité Théorique')

    plt.title(f"Simulation : Box-Muller General μ={mu}, σ={sigma}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    N = 5693  # Taille echantillon
    mu = 5
    sigma = 2
    samples = sim_box_muller_general(mu, sigma, N)
    analyze_and_plot_poisson(samples,mu,sigma)