import math

import matplotlib.pyplot as plt
import numpy as np


def sim_poisson(lmbd, N):
    """
    Simulate random variables from a Fish law using the inverse transform method.
    :param lmbd: Rate parameter of the Fish law
    :param N: Number of random variables to generate
    :return:
    """
    x = []
    for _ in range(N):
        u = np.random.rand()
        cumulative_prob = 0.0
        k = 0
        while True:
            # Formule : (e^-lmbd * lmbd^k) / k!
            prob = (math.exp(-lmbd) * (lmbd ** k)) / math.factorial(k)

            cumulative_prob += prob
            if cumulative_prob >= u:
                x.append(k)
                break
            k += 1
    return x


def density_poisson(k, lmbd):
    """
    density of poisson law
    :param k: Value at which to evaluate the density
    :param lmbd: Param of the Fish law
    :return: Density value at k
    """
    return (lmbd ** k) * math.exp(-lmbd) / math.factorial(k)


def analyze_and_plot_poisson(data, lmbd):
    """
    analyze (calc mean & var) and plot data
    theorical esperence for fish = lmbd
    theorical var for fish = lmbd
    :param data:
    :param lmbda:
    :param bins:
    :return:
    """
    mean_emp = np.mean(data)
    var_emp = np.var(data)
    print(f"Moyenne empirique : {mean_emp} espérance, {lmbd}")
    print(f"Variance empirique : {var_emp} Variance, {lmbd}")

    plt.figure(figsize=(8, 5))

    bins = max(data) + 2

    count, bins, ignored = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.arange(min(data), max(data) + 1)
    y_vals = [density_poisson(k, lmbd) for k in x_vals]
    plt.plot(x_vals, y_vals, 'ro', label='Théorie')

    plt.title(f"Simulation : Lois poisson λ={lmbd}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    lmbd = 8
    N = 5656
    samples = sim_poisson(lmbd, N)
    analyze_and_plot_poisson(samples, lmbd)
