"""Optuna optimizer wrapper using TPE, CMA-ES, and NSGA-II samplers."""

import numpy as np
import time
from typing import Dict, Any, List
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Wrapper for Optuna samplers.
    Maps GA->TPE, DE->CMA-ES, PSO->NSGA-II for fair comparison with PyMOO.
    """

    def __init__(self, pop_size: int = 50, n_gen: int = 100, seed: int = None):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.n_trials = n_gen  # match PyMOO generations

    def _create_objective(self, benchmark_problem):
        """Create Optuna objective from benchmark problem."""
        lb, ub = benchmark_problem.get_bounds()
        n_dim = benchmark_problem.n_dim

        def objective(trial):
            x = np.array([
                trial.suggest_float(f"x{i}", lb[i], ub[i])
                for i in range(n_dim)
            ])
            return benchmark_problem.evaluate(x)

        return objective

    def _run_optimization(
        self,
        sampler,
        benchmark_problem,
        algorithm_name: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run optimization with given sampler."""
        objective = self._create_objective(benchmark_problem)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        convergence_history = []

        def callback(study, trial):
            convergence_history.append(study.best_value)
            if verbose:
                print(f"    [Optuna] Trial {trial.number + 1}: Best = {study.best_value:.6e}")

        start_time = time.time()

        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[callback],
            show_progress_bar=False
        )

        elapsed_time = time.time() - start_time

        best_params = study.best_params
        best_solution = np.array([best_params[f"x{i}"] for i in range(benchmark_problem.n_dim)])

        return {
            "best_solution": best_solution,
            "best_fitness": float(study.best_value),
            "n_evaluations": len(study.trials),
            "convergence_history": convergence_history,
            "elapsed_time": elapsed_time,
            "algorithm": algorithm_name,
            "library": "Optuna"
        }

    def run_ga(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """Run GA-equivalent using TPE sampler."""
        sampler = TPESampler(
            seed=self.seed,
            n_startup_trials=self.pop_size,
            multivariate=True
        )

        return self._run_optimization(sampler, benchmark_problem, "TPE", verbose)

    def run_de(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """Run DE-equivalent using CMA-ES sampler."""
        sampler = CmaEsSampler(
            seed=self.seed,
            n_startup_trials=self.pop_size
        )

        return self._run_optimization(sampler, benchmark_problem, "CMA-ES", verbose)

    def run_pso(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """Run PSO-equivalent using NSGA-II sampler."""
        sampler = NSGAIISampler(
            seed=self.seed,
            population_size=self.pop_size
        )

        return self._run_optimization(sampler, benchmark_problem, "NSGA-II", verbose)

    def run_all(self, benchmark_problem, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run all algorithms on a problem."""
        return {
            "GA": self.run_ga(benchmark_problem, verbose),
            "DE": self.run_de(benchmark_problem, verbose),
            "PSO": self.run_pso(benchmark_problem, verbose)
        }


if __name__ == "__main__":
    from problems import Sphere, Rastrigin

    optimizer = OptunaOptimizer(pop_size=50, n_gen=100, seed=42)

    print("Testing Optuna Optimizer")
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
