"""
Visualization Module

This module provides plotting functions for visualizing comparison
results between PyMOO and Optuna optimization libraries.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import os


# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
COLORS = {
    "PyMOO": "#2E86AB",    # Blue
    "Optuna": "#E94F37",   # Red
}


def plot_convergence(
    convergence_data: Dict[str, List[List[float]]],
    problem_name: str,
    algorithm: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot convergence curves for both libraries.

    Args:
        convergence_data: Dictionary with 'PyMOO' and 'Optuna' histories
        problem_name: Name of the problem
        algorithm: Algorithm name
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for library, histories in convergence_data.items():
        if not histories:
            continue

        # Convert to numpy array, padding shorter histories
        max_len = max(len(h) for h in histories)
        padded = np.zeros((len(histories), max_len))

        for i, h in enumerate(histories):
            padded[i, :len(h)] = h
            if len(h) < max_len:
                padded[i, len(h):] = h[-1] if h else 0

        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)
        generations = np.arange(1, max_len + 1)

        color = COLORS.get(library, "#333333")
        ax.plot(generations, mean_curve, label=library, color=color, linewidth=2)
        ax.fill_between(
            generations,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color
        )

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title(f'{algorithm} Convergence on {problem_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_comparison_bars(
    results_df,
    metric: str = "Best_Fitness_Mean",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar charts comparing performance metrics.

    Args:
        results_df: DataFrame with experiment results
        metric: Column name for the metric to compare
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    problems = results_df['Problem'].unique()
    algorithms = results_df['Algorithm'].unique()

    n_problems = len(problems)
    n_algorithms = len(algorithms)

    fig, axes = plt.subplots(1, n_problems, figsize=(5 * n_problems, 6))

    if n_problems == 1:
        axes = [axes]

    bar_width = 0.35

    for ax, problem in zip(axes, problems):
        problem_data = results_df[results_df['Problem'] == problem]

        x = np.arange(n_algorithms)

        pymoo_vals = []
        optuna_vals = []

        for algo in algorithms:
            algo_data = problem_data[problem_data['Algorithm'] == algo]
            pymoo_val = algo_data[algo_data['Library'] == 'PyMOO'][metric].values
            optuna_val = algo_data[algo_data['Library'] == 'Optuna'][metric].values

            pymoo_vals.append(pymoo_val[0] if len(pymoo_val) > 0 else 0)
            optuna_vals.append(optuna_val[0] if len(optuna_val) > 0 else 0)

        bars1 = ax.bar(x - bar_width/2, pymoo_vals, bar_width,
                       label='PyMOO', color=COLORS['PyMOO'])
        bars2 = ax.bar(x + bar_width/2, optuna_vals, bar_width,
                       label='Optuna', color=COLORS['Optuna'])

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=12)
        ax.set_title(f'{problem}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.set_yscale('log')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_time_comparison(
    results_df,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap comparing execution times.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, library in zip(axes, ['PyMOO', 'Optuna']):
        lib_data = results_df[results_df['Library'] == library]

        pivot = lib_data.pivot(
            index='Problem',
            columns='Algorithm',
            values='Time_Mean'
        )

        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.3f}s',
                              ha="center", va="center", color="black", fontsize=10)

        ax.set_title(f'{library} Execution Time (seconds)', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Time (s)')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_overall_comparison(
    results_df,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create an overall comparison radar/spider chart.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    # Aggregate scores
    pymoo_data = results_df[results_df['Library'] == 'PyMOO']
    optuna_data = results_df[results_df['Library'] == 'Optuna']

    categories = ['Avg Fitness\n(lower is better)',
                  'Consistency\n(lower std)',
                  'Speed\n(lower time)',
                  'Best Result\n(lower min)']

    # Normalize scores (lower is better for all)
    pymoo_scores = [
        pymoo_data['Best_Fitness_Mean'].mean(),
        pymoo_data['Best_Fitness_Std'].mean(),
        pymoo_data['Time_Mean'].mean(),
        pymoo_data['Best_Fitness_Min'].mean()
    ]

    optuna_scores = [
        optuna_data['Best_Fitness_Mean'].mean(),
        optuna_data['Best_Fitness_Std'].mean(),
        optuna_data['Time_Mean'].mean(),
        optuna_data['Best_Fitness_Min'].mean()
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, pymoo_scores, width, label='PyMOO', color=COLORS['PyMOO'])
    bars2 = ax.bar(x + width/2, optuna_scores, width, label='Optuna', color=COLORS['Optuna'])

    ax.set_ylabel('Score (lower is better)')
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_all_figures(
    results_df,
    convergence_data: Dict,
    output_dir: str = "report/figures"
) -> List[str]:
    """
    Generate all figures for the report.

    Args:
        results_df: DataFrame with experiment results
        convergence_data: Dictionary with convergence histories
        output_dir: Directory to save figures

    Returns:
        List of saved figure paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # 1. Comparison bars
    path = os.path.join(output_dir, "comparison_bars.png")
    plot_comparison_bars(results_df, save_path=path)
    saved_paths.append(path)
    plt.close()

    # 2. Time comparison
    path = os.path.join(output_dir, "time_comparison.png")
    plot_time_comparison(results_df, save_path=path)
    saved_paths.append(path)
    plt.close()

    # 3. Overall comparison
    path = os.path.join(output_dir, "overall_comparison.png")
    plot_overall_comparison(results_df, save_path=path)
    saved_paths.append(path)
    plt.close()

    # 4. Convergence plots for each problem-algorithm combination
    for (problem, algorithm), data in convergence_data.items():
        path = os.path.join(output_dir, f"convergence_{problem.lower()}_{algorithm.lower()}.png")
        plot_convergence(data, problem, algorithm, save_path=path)
        saved_paths.append(path)
        plt.close()

    return saved_paths


if __name__ == "__main__":
    # Quick test with dummy data
    import pandas as pd

    test_data = {
        'Problem': ['Sphere', 'Sphere', 'Sphere', 'Sphere'],
        'Algorithm': ['GA', 'GA', 'DE', 'DE'],
        'Library': ['PyMOO', 'Optuna', 'PyMOO', 'Optuna'],
        'Best_Fitness_Mean': [1e-5, 2e-5, 1e-6, 3e-6],
        'Best_Fitness_Std': [1e-6, 2e-6, 1e-7, 3e-7],
        'Best_Fitness_Min': [1e-6, 1e-6, 1e-7, 1e-7],
        'Time_Mean': [0.5, 0.4, 0.6, 0.5],
        'Time_Std': [0.05, 0.04, 0.06, 0.05],
    }

    df = pd.DataFrame(test_data)

    print("Testing visualization module...")
    fig = plot_comparison_bars(df)
    plt.show()
