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

def gaussian_density(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)

def analyze_and_plot_poisson(data):
    """
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

    count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.linspace(min(bins), max(bins), 200)
    y_vals = [gaussian_density(x) for x in x_vals]
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Densité Théorique')

    plt.title(f"Simulation : Box-Muller Standard")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    N = 5693  # Taille echantillon
    samples = sim_box_muller_standard(N)
    analyze_and_plot_poisson(samples)