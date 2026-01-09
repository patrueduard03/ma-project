# Metaheuristic Algorithms Project: PyMOO vs Optuna

**Course**: Metaheuristic Algorithms  
**Year**: 2026

---

## What is this?

This is a comparative study of two optimization libraries: **PyMOO** and **Optuna**.
We tested them to see which one is better for standard mathematical problems like Sphere, Rastrigin, Ackley, and Rosenbrock.

Our goal was to see if Optuna (made for Machine Learning) can beat PyMOO (made for Evolutionary Algorithms) at its own game.

## How to Run It

The easiest way is to use Docker so you don't have to install anything manually.

### Using Docker (Recommended)

1.  **Fast Test** (Runs in ~1 min):
    ```bash
    docker compose run --rm quick-test
    ```

2.  **Full Experiment** (Runs in ~5 mins):
    ```bash
    docker compose run --rm experiment
    ```

This will run all the algorithms, save the results, and generate the plots in the `report/figures` folder.

### Manual Run (If you don't have Docker)

You need Python 3.10+ installed.

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the main script:
    ```bash
    python main.py
    ```

## The Report & Presentation

All the files you need for the final submission are in the `report/` folder:

*   **`report/main.tex`**: The full LaTeX report. It automatically pulls in the figures we generate.
*   **`report/presentation.tex`**: The 10-minute Beamer presentation slides.
*   **`report/references.bib`**: The bibliography with all the papers we cited.

To view them, just upload the whole `report/` folder to Overleaf and hit Compile.

## What's Inside?

*   `main.py`: The script that runs everything.
*   `src/`: The code for the algorithms and problems.
*   `report/figures/`: Where the charts and tables go after you run the code.

---

**Authors**:
Metaheuristic Algorithms Team
Master's in Computer Science
West University of Timișoara
