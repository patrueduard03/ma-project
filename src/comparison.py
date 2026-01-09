"""
Comparison Utilities Module

This module provides utilities for running comparison experiments
between PyMOO and Optuna optimization libraries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time

from src.problems import get_all_problems, BenchmarkProblem
from src.pymoo_optimizer import PymooOptimizer
from src.optuna_optimizer import OptunaOptimizer


class ComparisonRunner:
    """
    Runs comparison experiments between PyMOO and Optuna.

    Executes multiple runs of various algorithms on benchmark problems
    and collects statistical results.
    """

    def __init__(self, pop_size: int = 50, n_gen: int = 100, n_runs: int = 10):
        """
        Initialize the comparison runner.

        Args:
            pop_size: Population size for all algorithms
            n_gen: Number of generations
            n_runs: Number of independent runs for statistical analysis
        """
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_runs = n_runs
        self.results = {}

    def run_single_comparison(
        self,
        problem: BenchmarkProblem,
        algorithm: str,
        seed: int = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a single comparison between PyMOO and Optuna.

        Args:
            problem: BenchmarkProblem instance
            algorithm: Algorithm name ('GA', 'DE', or 'PSO')
            seed: Random seed

        Returns:
            Dictionary with results from both libraries
        """
        pymoo_opt = PymooOptimizer(self.pop_size, self.n_gen, seed)
        optuna_opt = OptunaOptimizer(self.pop_size, self.n_gen, seed)

        run_method = f"run_{algorithm.lower()}"

        pymoo_result = getattr(pymoo_opt, run_method)(problem)
        optuna_result = getattr(optuna_opt, run_method)(problem)

        return {
            "PyMOO": pymoo_result,
            "Optuna": optuna_result
        }

    def run_experiment(
        self,
        problems: Dict[str, BenchmarkProblem] = None,
        algorithms: List[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run a full comparison experiment with multiple runs.

        Args:
            problems: Dictionary of problems to test (default: all)
            algorithms: List of algorithms to use (default: ['GA', 'DE', 'PSO'])
            verbose: Whether to print progress

        Returns:
            DataFrame with experiment results
        """
        if problems is None:
            problems = get_all_problems(n_dim=10)

        if algorithms is None:
            algorithms = ['GA', 'DE', 'PSO']

        all_results = []

        total_experiments = len(problems) * len(algorithms) * self.n_runs * 2
        current = 0

        for prob_name, problem in problems.items():
            for algo in algorithms:
                pymoo_fitness = []
                pymoo_times = []
                optuna_fitness = []
                optuna_times = []

                pymoo_histories = []
                optuna_histories = []

                for run in range(self.n_runs):
                    seed = 42 + run

                    if verbose:
                        current += 2
                        print(f"\r[{current}/{total_experiments}] "
                              f"{prob_name} - {algo} - Run {run + 1}/{self.n_runs}", end="")

                    comparison = self.run_single_comparison(problem, algo, seed)

                    pymoo_fitness.append(comparison["PyMOO"]["best_fitness"])
                    pymoo_times.append(comparison["PyMOO"]["elapsed_time"])
                    pymoo_histories.append(comparison["PyMOO"]["convergence_history"])

                    optuna_fitness.append(comparison["Optuna"]["best_fitness"])
                    optuna_times.append(comparison["Optuna"]["elapsed_time"])
                    optuna_histories.append(comparison["Optuna"]["convergence_history"])

                # Store PyMOO results
                all_results.append({
                    "Problem": prob_name,
                    "Algorithm": algo,
                    "Library": "PyMOO",
                    "Best_Fitness_Mean": np.mean(pymoo_fitness),
                    "Best_Fitness_Std": np.std(pymoo_fitness),
                    "Best_Fitness_Min": np.min(pymoo_fitness),
                    "Time_Mean": np.mean(pymoo_times),
                    "Time_Std": np.std(pymoo_times),
                    "Global_Minimum": problem.global_minimum
                })

                # Store Optuna results
                all_results.append({
                    "Problem": prob_name,
                    "Algorithm": algo,
                    "Library": "Optuna",
                    "Best_Fitness_Mean": np.mean(optuna_fitness),
                    "Best_Fitness_Std": np.std(optuna_fitness),
                    "Best_Fitness_Min": np.min(optuna_fitness),
                    "Time_Mean": np.mean(optuna_times),
                    "Time_Std": np.std(optuna_times),
                    "Global_Minimum": problem.global_minimum
                })

                # Store convergence histories for later visualization
                self.results[(prob_name, algo)] = {
                    "PyMOO": pymoo_histories,
                    "Optuna": optuna_histories
                }

        if verbose:
            print("\nExperiment completed!")

        return pd.DataFrame(all_results)

    def generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary comparison table.

        Args:
            df: DataFrame from run_experiment

        Returns:
            Pivot table comparing libraries across problems and algorithms
        """
        summary = df.pivot_table(
            index=['Problem', 'Algorithm'],
            columns='Library',
            values=['Best_Fitness_Mean', 'Best_Fitness_Std', 'Time_Mean'],
            aggfunc='first'
        )

        return summary

    def get_convergence_data(
        self,
        problem_name: str,
        algorithm: str
    ) -> Dict[str, List[List[float]]]:
        """
        Get convergence history data for visualization.

        Args:
            problem_name: Name of the problem
            algorithm: Algorithm name

        Returns:
            Dictionary with convergence histories for both libraries
        """
        key = (problem_name, algorithm)
        if key in self.results:
            return self.results[key]
        return {"PyMOO": [], "Optuna": []}


def format_results_table(df: pd.DataFrame) -> str:
    """
    Format results DataFrame as a nice table string.

    Args:
        df: Results DataFrame

    Returns:
        Formatted table string
    """
    from tabulate import tabulate

    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Problem'],
            row['Algorithm'],
            row['Library'],
            f"{row['Best_Fitness_Mean']:.4e} ± {row['Best_Fitness_Std']:.2e}",
            f"{row['Time_Mean']:.3f}s"
        ])

    headers = ['Problem', 'Algorithm', 'Library', 'Best Fitness (Mean ± Std)', 'Time']
    return tabulate(table_data, headers=headers, tablefmt='grid')


if __name__ == "__main__":
    # Run a quick comparison test
    print("Running Comparison Test")
    print("=" * 60)

    runner = ComparisonRunner(pop_size=30, n_gen=50, n_runs=3)

    # Use only Sphere for quick test
    from src.problems import Sphere
    test_problems = {"Sphere": Sphere(n_dim=10)}

    df = runner.run_experiment(
        problems=test_problems,
        algorithms=['GA', 'DE'],
        verbose=True
    )

    print("\n" + format_results_table(df))
