import matplotlib.pyplot as plt
import numpy as np
import math


def sim_binomiale(n, p, N):
    """
    Simulate random variables from a binomial law using the inverse transform method.
    :param n: Number of trials
    :param p: Probability of success on each trial
    :param N: Number of random variables to generate
    :return:
    """
    x = []

    for _ in range(N):
        u = np.random.rand()
        cumulative_prob = 0.0
        k = 0
        while k <= n:
            # prob = (n, k) * p^k * (1-p)^(n-k)
            prob = (math.comb(n, k)) * (p ** k) * ((1 - p) ** (n - k))
            cumulative_prob += prob
            if cumulative_prob >= u:
                x.append(k)
                break
            k += 1

    return x


def density_binomiale(k, n, p):
    """
    density of binomial law
    :param k: Value at which to evaluate the density
    :param n: Number of trials
    :param p: Probability of success on each trial
    :return: Density value at k
    """
    return (math.comb(n, k)) * (p ** k) * ((1 - p) ** (n - k))


def analyze_and_plot_binomiale(data, n, p):
    """
    analyze (calc mean & var) and plot data
    theorical esperence for bino = np
    theorical var for bino = np(1-p)
    :param data:
    :param lmbda:
    :param bins:
    :return:
    """
    mean_emp = np.mean(data)
    var_emp = np.var(data)
    print(f"Moyenne empirique : {mean_emp} espérance, {n * p}")
    print(f"Variance empirique : {var_emp} Variance, {n * p * (1 - p)}")

    plt.figure(figsize=(8, 5))

    bins = range(0, n+2)

    count, bins, ignored = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.arange(min(data), max(data) + 1)
    y_vals = [density_binomiale(k, n, p) for k in x_vals]
    plt.plot(x_vals, y_vals, 'ro', label='Théorie')

    plt.title(f"Simulation : Lois binomiale n={n}, p={p}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    p = 0.69
    n = 10
    N = 5658
    samples = sim_binomiale(n, p, N)
    analyze_and_plot_binomiale(samples, n, p)
