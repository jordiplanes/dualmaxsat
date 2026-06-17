# DualMaxSAT: Implicit Hitting Set with Branch-and-Price Techniques

This repository implements an **Implicit Hitting Set (IHS)** MaxSAT solver and explores its connection to **Branch-and-Price (BnP)** and **Dantzig-Wolfe Decomposition**.

## Overview

IHS solvers alternate between a SAT oracle (to find unsatisfiable cores) and an IP solver (to find a minimum-cost hitting set of the cores). This repository treats IHS as a decomposition method from Operations Research, specifically column generation on the dual packing LP.

## Key Features

The solver (`src/ihs.py`) implements several "Transferable Techniques" from OR to MaxSAT:

- **Multi-core Extraction:** Finding multiple disjoint cores per master solve to reduce iterations.
- **Reduced Cost Fixing:** Hardening soft clauses based on LP dual prices and reduced costs.
- **Pareto-Optimal Core Selection:** Biasing the SAT solver toward cores that maximize violation.
- **Stabilization:** Wentges-style box penalties and in-out separation to damp dual oscillation.
- **Warm-Starting:** Initializing the hitting set master with cores harvested from a Fu-Malik-style core-guided solver.

## Usage

Ensure you have a virtual environment with the necessary dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the toy instances example:

```bash
python3 src/example.py
```

Run the parameterized benchmarks:

```bash
python3 src/benchmarks.py
```

## Documentation

The theoretical foundations are detailed in the included memo (`memo.pdf` / `memo.tex`), which is divided into four parts:
1. **Decomposition View:** IHS as Dantzig-Wolfe and Core-guided as Benders.
2. **Equivalence Analysis:** Exact equivalence at the LP level and the integrality gap.
3. **Taxonomy:** Mapping IHS to Lagrangian relaxation and cutting planes.
4. **Empirical Study:** Analysis of design axes in IHS for weighted CSPs.
