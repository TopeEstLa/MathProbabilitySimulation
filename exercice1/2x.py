import matplotlib.pyplot as plt
import numpy as np

def sim_2x(n):
    """
    On a F(x) = 2x = u  <=> x = sqrt(u)
    Simulate random variables from an exponential law using the inverse transform method.
    :param n: Number of random variables to generate
    :return:
    """
    x = []

    for _ in range(n):
        u = np.random.rand()
        x.append(np.sqrt(u))

    return x

def density_2x(x):
    """
    :param x: Value at which to evaluate the density
    :return: Density value at x
    """
    return 2 * x

def analyze_and_plot_2x(data, bins=30):
    """

    :param data:
    :param bins:
    :return:
    """
    mean_emp = np.mean(data)
    var_emp = np.var(data)
    print(f"Moyenne empirique : {mean_emp} espérance, {0.6667}")
    print(f"Variance empirique : {var_emp} Variance, {0.0556}")

    plt.figure(figsize=(8, 5))
    count, bins, ignored = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black',
                                    label='Valeur Simulation')

    x_vals = np.linspace(min(bins), max(bins), 200)
    y_vals = [density_2x(x) for x in x_vals]
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Densité Théorique')

    plt.title(f"Simulation : f(x) = 2x")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    n = 5659
    samples = sim_2x(n)
    analyze_and_plot_2x(samples)