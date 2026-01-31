"""
Optuna Optimizer Module

This module provides wrapper classes for Optuna optimization algorithms.
Optuna is a hyperparameter optimization framework that provides various
samplers for efficient optimization.

"""

import numpy as np
import time
from typing import Dict, Any, List
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler


# Suppress Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Wrapper class for Optuna optimization algorithms.

    Provides a unified interface for running TPE, CMA-ES, and NSGA-II samplers
    with consistent parameter handling and result collection.
    
    Algorithm mapping:
    - GA (Genetic Algorithm) -> TPESampler (Tree-structured Parzen Estimator)
    - DE (Differential Evolution) -> CmaEsSampler (Covariance Matrix Adaptation)
    - PSO (Particle Swarm Optimization) -> NSGAIISampler (NSGA-II based sampling)
    """

    def __init__(self, pop_size: int = 50, n_gen: int = 100, seed: int = None):
        """
        Initialize the optimizer.

        Args:
            pop_size: Population size (used for sampler configuration)
            n_gen: Number of generations (total trials = pop_size * n_gen)
            seed: Random seed for reproducibility
        """
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.n_trials = n_gen  # Match PyMOO generations for fair comparison

    def _create_objective(self, benchmark_problem):
        """
        Create Optuna objective function from benchmark problem.

        Args:
            benchmark_problem: BenchmarkProblem instance

        Returns:
            Optuna objective function
        """
        lb, ub = benchmark_problem.get_bounds()
        n_dim = benchmark_problem.n_dim

        def objective(trial):
            # Create solution vector by suggesting values for each dimension
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
        """
        Run an optimization with the given sampler and collect results.

        Args:
            sampler: Optuna sampler instance
            benchmark_problem: BenchmarkProblem instance
            algorithm_name: Name of the algorithm for reporting
            verbose: Whether to print progress

        Returns:
            Dictionary with optimization results
        """
        objective = self._create_objective(benchmark_problem)

        # Create study with the sampler
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        # Track convergence history
        convergence_history = []

        def callback(study, trial):
            convergence_history.append(study.best_value)
            if verbose:
                print(f"    [Optuna] Trial {trial.number + 1}: Best = {study.best_value:.6e}")

        start_time = time.time()

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[callback],
            show_progress_bar=False
        )

        elapsed_time = time.time() - start_time

        # Extract best solution
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
        """
        Run GA-equivalent optimization using TPE Sampler.

        TPE (Tree-structured Parzen Estimator) provides efficient
        sampling similar to genetic algorithm behavior.

        Args:
            benchmark_problem: BenchmarkProblem instance
            verbose: Whether to print progress

        Returns:
            Optimization results dictionary
        """
        sampler = TPESampler(
            seed=self.seed,
            n_startup_trials=self.pop_size,  # Initial random sampling
            multivariate=True
        )

        return self._run_optimization(sampler, benchmark_problem, "TPE", verbose)

    def run_de(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """
        Run DE-equivalent optimization using CMA-ES Sampler.

        CMA-ES (Covariance Matrix Adaptation Evolution Strategy) provides
        similar behavior to Differential Evolution.

        Args:
            benchmark_problem: BenchmarkProblem instance
            verbose: Whether to print progress

        Returns:
            Optimization results dictionary
        """
        sampler = CmaEsSampler(
            seed=self.seed,
            n_startup_trials=self.pop_size
        )

        return self._run_optimization(sampler, benchmark_problem, "CMA-ES", verbose)

    def run_pso(self, benchmark_problem, verbose: bool = False) -> Dict[str, Any]:
        """
        Run PSO-equivalent optimization using NSGA-II Sampler.

        NSGA-II provides population-based optimization similar to
        Particle Swarm Optimization.

        Args:
            benchmark_problem: BenchmarkProblem instance
            verbose: Whether to print progress

        Returns:
            Optimization results dictionary
        """
        sampler = NSGAIISampler(
            seed=self.seed,
            population_size=self.pop_size
        )

        return self._run_optimization(sampler, benchmark_problem, "NSGA-II", verbose)

    def run_all(self, benchmark_problem, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Run all available algorithms on a problem.

        Args:
            benchmark_problem: BenchmarkProblem instance
            verbose: Whether to print progress

        Returns:
            Dictionary mapping algorithm names to results
        """
        return {
            "GA": self.run_ga(benchmark_problem, verbose),
            "DE": self.run_de(benchmark_problem, verbose),
            "PSO": self.run_pso(benchmark_problem, verbose)
        }


if __name__ == "__main__":
    # Test the Optuna optimizer
    from problems import Sphere, Rastrigin

    optimizer = OptunaOptimizer(pop_size=50, n_gen=100, seed=42)

    print("Testing Optuna Optimizer")
    print("=" * 50)

    # Test on Sphere function
    sphere = Sphere(n_dim=10)
    results = optimizer.run_all(sphere, verbose=False)

    print(f"\nSphere Function (10D):")
    for algo_name, result in results.items():
        print(f"  {algo_name}: Best = {result['best_fitness']:.6e}, "
              f"Time = {result['elapsed_time']:.3f}s")

    # Test on Rastrigin function
    rastrigin = Rastrigin(n_dim=10)
    results = optimizer.run_all(rastrigin, verbose=False)

    print(f"\nRastrigin Function (10D):")
    for algo_name, result in results.items():
        print(f"  {algo_name}: Best = {result['best_fitness']:.6e}, "
              f"Time = {result['elapsed_time']:.3f}s")
