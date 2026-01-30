# Metaheuristic Algorithms Project: PyMOO vs Optuna

**Course**: Metaheuristic Algorithms  
**Year**: 2026

---

## What is this?

This is a comparative study of two optimization libraries: **PyMOO** and **Optuna**.
We tested them to see which one is better for standard mathematical problems like Sphere, Rastrigin, Ackley, and Rosenbrock.

Our goal was to see if Optuna (made for Machine Learning) can beat PyMOO (made for Evolutionary Algorithms) at its own game.

## How to Run It

### Option 1: Using Docker

If you have Docker installed, this is the easiest way:

```bash
# Quick test (~1 min)
docker compose run --rm quick-test

# Full experiment (~5 mins)
docker compose run --rm experiment
```

### Option 2: Running Locally (No Docker)

If you don't have Docker, no problem. Just make sure you have Python 3.10 or newer.

```bash
# First, create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt

# Run a quick test to make sure everything works
python main.py --quick

# Or run the full experiment
python main.py
```

The script will print the results in the terminal and save figures to `report/figures/`.

You can also tweak the parameters if you want:
```bash
python main.py --pop-size 50 --generations 100 --runs 10
```

## Project Structure

- `main.py` - Runs everything
- `src/` - Algorithm implementations and benchmark problems
- `report/figures/` - Generated charts and tables

---

**Authors**: Metaheuristic Algorithms Team  
Master's in Computer Science, West University of Timișoara
