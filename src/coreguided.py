"""Minimal Fu-Malik-style core-guided MaxSAT.

Used solely as a warm-start producer for IHS: we run it for a small
iteration budget and harvest the cores it discovers.
"""

from pysat.formula import WCNF
from pysat.solvers import Solver


def harvest_cores(wcnf: WCNF, budget: int, sat_name: str = "g3") -> list[frozenset[int]]:
    """Run Fu-Malik on wcnf for at most `budget` iterations; return discovered cores.

    Cores are returned as frozensets of soft-clause indices into wcnf.soft.
    """
    n_soft = len(wcnf.soft)
    if n_soft == 0:
        return []

    base_var = max(
        wcnf.nv,
        max((abs(l) for cl in wcnf.soft + wcnf.hard for l in cl), default=0),
    )
    selectors = [base_var + 1 + i for i in range(n_soft)]

    solver = Solver(name=sat_name)
    for cl in wcnf.hard:
        solver.add_clause(cl)
    for sel, soft_cl in zip(selectors, wcnf.soft):
        solver.add_clause(list(soft_cl) + [-sel])

    sel_to_idx = {sel: i for i, sel in enumerate(selectors)}
    active = set(selectors)
    cores: list[frozenset[int]] = []

    for _ in range(budget):
        assumptions = sorted(active)
        if solver.solve(assumptions=assumptions):
            break
        raw_core = solver.get_core() or []
        core_sels = [lit for lit in raw_core if lit in sel_to_idx]
        if not core_sels:
            break
        cores.append(frozenset(sel_to_idx[s] for s in core_sels))
        # Drop the lowest-numbered selector from `active` to make progress;
        # the goal here is variety of cores, not bound-correctness.
        active.discard(core_sels[0])

    solver.delete()
    return cores
