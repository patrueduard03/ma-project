"""
Main runner for PyMOO vs Optuna comparison.
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problems import get_all_problems
from src.comparison import ComparisonRunner, format_results_table
from src.visualization import create_all_figures


def run_full_experiment(
    pop_size: int = 50,
    n_gen: int = 50,
    n_runs: int = 5,
    n_dim: int = 10,
    output_dir: str = "report/figures"
) -> pd.DataFrame:
    """Run the full comparison experiment."""
    print("=" * 70)
    print("  Comparative Study: PyMOO vs Optuna")
    print("  Metaheuristic Algorithms Course Project 2025-2026")
    print("=" * 70)

    print(f"\nExperiment Configuration:")
    print(f"  - Population Size: {pop_size}")
    print(f"  - Generations: {n_gen}")
    print(f"  - Independent Runs: {n_runs}")
    print(f"  - Problem Dimensions: {n_dim}")
    print(f"  - Output Directory: {output_dir}")

    problems = get_all_problems(n_dim=n_dim)
    print(f"\nBenchmark Problems: {list(problems.keys())}")

    runner = ComparisonRunner(
        pop_size=pop_size,
        n_gen=n_gen,
        n_runs=n_runs
    )

    print("\n" + "-" * 70)
    print("Running Optimization Experiments...")
    print("-" * 70)

    results_df = runner.run_experiment(
        problems=problems,
        algorithms=['GA', 'DE', 'PSO'],
        verbose=True
    )

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(format_results_table(results_df))

    print("\n" + "-" * 70)
    print("Generating Visualization Figures...")
    print("-" * 70)

    saved_figures = create_all_figures(
        results_df,
        runner.results,
        output_dir=output_dir
    )

    print(f"\nSaved {len(saved_figures)} figures:")
    for fig_path in saved_figures:
        print(f"  - {fig_path}")

    csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    latex_table = generate_latex_table(results_df)
    latex_path = os.path.join(output_dir, "results_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    print("\n" + "=" * 70)
    print("  Experiment Complete!")
    print("=" * 70)

    return results_df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from results."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparative Results: PyMOO vs Optuna}
\label{tab:results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllrr}
\toprule
\textbf{Problem} & \textbf{Algorithm} & \textbf{Library} & \textbf{Best Fitness} & \textbf{Std Dev} & \textbf{Time (s)} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Problem']} & {row['Algorithm']} & {row['Library']} & "
        latex += f"{row['Best_Fitness_Mean']:.4e} & {row['Best_Fitness_Std']:.2e} & "
        latex += f"{row['Time_Mean']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

    return latex


def run_quick_test():
    """Quick test with reduced parameters."""
    print("Running Quick Test (reduced parameters)...")

    from src.problems import Sphere, Rastrigin

    runner = ComparisonRunner(pop_size=30, n_gen=50, n_runs=3)

    problems = {
        "Sphere": Sphere(n_dim=10),
        "Rastrigin": Rastrigin(n_dim=10)
    }

    df = runner.run_experiment(
        problems=problems,
        algorithms=['GA', 'DE'],
        verbose=True
    )

    print("\n" + format_results_table(df))

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PyMOO vs Optuna Comparative Study'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a quick test with reduced parameters'
    )

    parser.add_argument(
        '--pop-size',
        type=int,
        default=50,
        help='Population size (default: 50)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Number of generations (default: 100)'
    )

    parser.add_argument(
        '--runs',
        type=int,
        default=10,
        help='Number of independent runs (default: 10)'
    )

    parser.add_argument(
        '--dimensions',
        type=int,
        default=10,
        help='Problem dimensions (default: 10)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='report/figures',
        help='Output directory for figures (default: report/figures)'
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_full_experiment(
            pop_size=args.pop_size,
            n_gen=args.generations,
            n_runs=args.runs,
            n_dim=args.dimensions,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
