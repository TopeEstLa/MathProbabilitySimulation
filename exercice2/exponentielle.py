import matplotlib.pyplot as plt
import numpy as np


def sim_expo(lmbda, n):
    """
    F(x) = 1 - exp(-lambda*x) = u  <=> x = -ln(1-u)/lambda
    Simulate random variables from an exponential law using the inverse transform method.
    :param lmbda:  Rate parameter of the exponential law
    :param n: Number of random variables to generate
    :return:
    """
    x = []

    for _ in range(n):
        u = np.random.rand()
        x.append(-np.log(1 - u) / lmbda)

    return x


def sim_tcl_expo(n, N, lmbd):
    """
    Simulate random variables from the TCL of an exponential law using the inverse transform method.
    :param n:
    :param N:
    :param lmbd:
    :return:
    """
    mu = 1 / lmbd
    sigma = 1 / lmbd

    samples = []
    for _ in range(N):
        sum_x = np.sum(sim_expo(lmbd, n))
        z = (sum_x - n * mu) / (sigma * np.sqrt(n))
        samples.append(z)
    return np.array(samples)


def gaussian_density(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def analyze_and_plot_poisson(data, lmbd):
    """
    analyze (calc mean & var) and plot data
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

    plt.title(f"Simulation : TCL (Exponentielle λ={lmbd})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    N = 10000  # Taille echantillon
    n = 100
    lmbd = 7
    samples = sim_tcl_expo(n, N, lmbd)
    analyze_and_plot_poisson(samples, lmbd)
