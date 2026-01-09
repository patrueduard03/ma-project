"""
Benchmark Optimization Problems

This module defines standard benchmark functions used for testing
optimization algorithms. Each function has a known global minimum
which allows for objective comparison of algorithm performance.
"""

import numpy as np


class BenchmarkProblem:
    """Base class for benchmark optimization problems."""

    def __init__(self, n_dim: int = 10, bounds: tuple = None):
        """
        Initialize the benchmark problem.

        Args:
            n_dim: Number of dimensions
            bounds: Tuple of (lower_bound, upper_bound)
        """
        self.n_dim = n_dim
        self.bounds = bounds or (-5.12, 5.12)
        self.global_minimum = 0.0
        self.name = "Benchmark"

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function at point x."""
        raise NotImplementedError

    def get_bounds(self) -> tuple:
        """Return bounds as (lb_array, ub_array)."""
        lb = np.full(self.n_dim, self.bounds[0])
        ub = np.full(self.n_dim, self.bounds[1])
        return lb, ub


class Sphere(BenchmarkProblem):
    """
    Sphere Function (De Jong's Function 1)

    f(x) = sum(x_i^2)

    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^n

    Properties:
    - Unimodal
    - Convex
    - Continuous
    - Differentiable
    """

    def __init__(self, n_dim: int = 10):
        super().__init__(n_dim, bounds=(-5.12, 5.12))
        self.name = "Sphere"
        self.global_minimum = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)


class Rastrigin(BenchmarkProblem):
    """
    Rastrigin Function

    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))

    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^n

    Properties:
    - Highly multimodal
    - Regularly distributed local minima
    - Non-convex
    """

    def __init__(self, n_dim: int = 10):
        super().__init__(n_dim, bounds=(-5.12, 5.12))
        self.name = "Rastrigin"
        self.global_minimum = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


class Ackley(BenchmarkProblem):
    """
    Ackley Function

    f(x) = -20*exp(-0.2*sqrt(1/n * sum(x_i^2)))
           - exp(1/n * sum(cos(2*pi*x_i))) + 20 + e

    Global minimum: f(0,...,0) = 0
    Search domain: [-32.768, 32.768]^n

    Properties:
    - Multimodal
    - Nearly flat outer region
    - Large hole at center
    """

    def __init__(self, n_dim: int = 10):
        super().__init__(n_dim, bounds=(-32.768, 32.768))
        self.name = "Ackley"
        self.global_minimum = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))

        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)

        return term1 + term2 + 20 + np.e


class Rosenbrock(BenchmarkProblem):
    """
    Rosenbrock Function (Banana Function)

    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Global minimum: f(1,...,1) = 0
    Search domain: [-5, 10]^n

    Properties:
    - Unimodal (for n<=3)
    - Valley-shaped
    - Non-convex
    - Difficult to optimize due to narrow valley
    """

    def __init__(self, n_dim: int = 10):
        super().__init__(n_dim, bounds=(-5.0, 10.0))
        self.name = "Rosenbrock"
        self.global_minimum = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        result = 0.0
        for i in range(len(x) - 1):
            result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return result


def get_all_problems(n_dim: int = 10) -> dict:
    """
    Get all benchmark problems as a dictionary.

    Args:
        n_dim: Number of dimensions for each problem

    Returns:
        Dictionary mapping problem names to problem instances
    """
    return {
        "Sphere": Sphere(n_dim),
        "Rastrigin": Rastrigin(n_dim),
        "Ackley": Ackley(n_dim),
        "Rosenbrock": Rosenbrock(n_dim),
    }


if __name__ == "__main__":
    # Test the benchmark functions
    problems = get_all_problems(n_dim=10)

    for name, problem in problems.items():
        x_zero = np.zeros(problem.n_dim)
        x_ones = np.ones(problem.n_dim)

        print(f"\n{name} Function:")
        print(f"  Dimensions: {problem.n_dim}")
        print(f"  Bounds: {problem.bounds}")
        print(f"  f(0,...,0) = {problem.evaluate(x_zero):.6f}")
        print(f"  f(1,...,1) = {problem.evaluate(x_ones):.6f}")
        print(f"  Global minimum: {problem.global_minimum}")
