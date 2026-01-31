"""PyMOO optimizer wrapper for GA, DE, and PSO algorithms."""

import numpy as np
import time
from typing import Dict, Any, Callable

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class PymooProblem(Problem):
    """Wrapper to convert benchmark problems to PyMOO format."""

    def __init__(self, benchmark_problem):
        self.benchmark = benchmark_problem
        lb, ub = benchmark_problem.get_bounds()

        super().__init__(
            n_var=benchmark_problem.n_dim,
            n_obj=1,
            n_constr=0,
            xl=lb,
            xu=ub
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective for population."""
        out["F"] = np.array([self.benchmark.evaluate(xi) for xi in x])


class PymooOptimizer:
    """Wrapper for PyMOO optimization algorithms."""

    def __init__(self, pop_size: int = 50, n_gen: int = 100, seed: int = None):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.history = []

    def _run_algorithm(self, algorithm, problem, verbose: bool = False) -> Dict[str, Any]:
        """Run algorithm and collect results."""
        termination = get_termination("n_gen", self.n_gen)

        history = []

        class HistoryCallback:
            def __init__(self, verbose=False, problem_name=""):
                self.data = []
                self.verbose = verbose
                self.problem_name = problem_name

            def __call__(self, algorithm):
                if algorithm.opt is not None:
                    best_f = algorithm.opt.get("F").min()
                    self.data.append(best_f)
                    if self.verbose:
                        gen = len(self.data)
                        print(f"    [PyMOO] {self.problem_name} Epoch {gen}: Best = {best_f:.6e}")

        callback = HistoryCallback(verbose=verbose, problem_name=problem.benchmark.name)

        start_time = time.time()

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=self.seed,
            verbose=verbose,
            callback=callback
        )

        elapsed_time = time.time() - start_time

        return {
            "best_solution": result.X,
            "best_fitness": float(result.F[0]),
            "n_evaluations": result.algorithm.evaluator.n_eval,
            "convergence_history": callback.data,
            "elapsed_time": elapsed_time,
            "algorithm": algorithm.__class__.__name__,
            "library": "PyMOO"
        }

    def run_ga(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """Run Genetic Algorithm."""
        problem = PymooProblem(benchmark_problem)

        algorithm = GA(
            pop_size=self.pop_size,
            eliminate_duplicates=True
        )

        return self._run_algorithm(algorithm, problem, verbose)

    def run_de(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """Run Differential Evolution."""
        problem = PymooProblem(benchmark_problem)

        algorithm = DE(
            pop_size=self.pop_size,
            variant="DE/rand/1/bin",
            CR=0.9,
            F=0.8,
            dither="vector"
        )

        return self._run_algorithm(algorithm, problem, verbose)

    def run_pso(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """Run Particle Swarm Optimization."""
        problem = PymooProblem(benchmark_problem)

        algorithm = PSO(
            pop_size=self.pop_size,
            w=0.9,
            c1=2.0,
            c2=2.0
        )

        return self._run_algorithm(algorithm, problem, verbose)

    def run_all(self, benchmark_problem, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run all algorithms on a problem."""
        return {
            "GA": self.run_ga(benchmark_problem, verbose),
            "DE": self.run_de(benchmark_problem, verbose),
            "PSO": self.run_pso(benchmark_problem, verbose)
        }


if __name__ == "__main__":
    from problems import Sphere, Rastrigin

    optimizer = PymooOptimizer(pop_size=50, n_gen=100, seed=42)

    print("Testing PyMOO Optimizer")
    print("=" * 50)

    sphere = Sphere(n_dim=10)
    results = optimizer.run_all(sphere, verbose=False)

    print(f"\nSphere Function (10D):")
    for algo_name, result in results.items():
        print(f"  {algo_name}: Best = {result['best_fitness']:.6e}, "
              f"Time = {result['elapsed_time']:.3f}s")

    rastrigin = Rastrigin(n_dim=10)
    results = optimizer.run_all(rastrigin, verbose=False)

    print(f"\nRastrigin Function (10D):")
    for algo_name, result in results.items():
        print(f"  {algo_name}: Best = {result['best_fitness']:.6e}, "
              f"Time = {result['elapsed_time']:.3f}s")
