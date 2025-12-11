import matplotlib.pyplot as plt
import numpy as np


def sim_expo(lmbda, n):
    """
    On a F(x) = 1 - exp(-lambda*x) = u  <=> x = -ln(1-u)/lambda
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


def density_expo(x, lmbda):
    """
    :param x: Value at which to evaluate the density
    :param lmbda: Param of the exponential law
    :return: Density value at x
    """
    if x < 0:
        return 0
    return lmbda * np.exp(-lmbda * x)


def analyze_and_plot_expo(data, lmbda, bins=30):
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
    print(f"Moyenne empirique : {mean_emp} espérance, {1 / lmbda}")
    print(f"Variance empirique : {var_emp} Variance, {1 / (lmbda ** 2)}")

    plt.figure(figsize=(8, 5))
    count, bins, ignored = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.linspace(min(bins), max(bins), 200)
    y_vals = [density_expo(x, lmbda) for x in x_vals]
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Densité Théorique')

    plt.title(f"Simulation : Lois expo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    lmbda = 7
    n = 5693
    samples = sim_expo(lmbda, n)
    analyze_and_plot_expo(samples, lmbda)
