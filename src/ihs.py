"""Implicit Hitting Set MaxSAT solver with transferable OR techniques.

The base solver implements Algorithm 2 of the report. Each technique from
Section 7 (Transferable Techniques from OR to MaxSAT) is exposed as an
option on `IHSConfig`:

  - warm_start_cores         : cross-family warm-starting (§7.1)
  - pareto_selection         : Pareto-optimal core selection (§7.2)
  - wentges_alpha0, decay    : Wentges-style box stabilization (§7.3)
  - in_out_lambda            : in-out separation (§7.3)
  - price_aware_order        : price-aware assumption ordering (§7.4)

Toggling all of them together corresponds to the hybrid primal-dual
sketch in §7.5.

The pricing-side program of Part 2 (§ "Closing the Gap") adds:

  - threshold_sweep          : price-guided separation against the
                               fractional master prices ȳ (binary search
                               over price levels; heuristic separator)
  - auto_abstraction /       : abstract cores à la Berg-Bacchus, read as
    abstraction_sets           column aggregation: counting literals over
                               equal-weight clusters yield rank-k master
                               rows Σ_{i∈A} y_i ≥ k that can close part of
                               the LP-IP integrality gap
"""

from dataclasses import dataclass, field
from typing import Optional

import pulp
from pysat.card import ITotalizer
from pysat.formula import WCNF
from pysat.solvers import Solver


@dataclass
class IHSConfig:
    sat_name: str = "g3"
    warm_start_cores: list = field(default_factory=list)
    pareto_selection: bool = False
    wentges_alpha0: float = 0.0
    wentges_decay: float = 0.5
    in_out_lambda: float = 0.0
    price_aware_order: bool = False
    multi_core: bool = False
    reduced_cost_fixing: bool = False
    threshold_sweep: bool = False
    # Abstraction sets for abstract cores: either an explicit list of
    # index collections (each with uniform weight), or auto-clustering of
    # equal-weight soft clauses into sets of size >= 3.
    abstraction_sets: Optional[list] = None
    auto_abstraction: bool = False
    max_iters: int = 1000


@dataclass
class IHSStats:
    iterations: int = 0
    cores_found: int = 0
    sat_calls: int = 0
    ip_solves: int = 0
    pareto_refinements: int = 0
    abstract_rows: int = 0
    sweep_calls: int = 0
    lp_value: Optional[float] = None
    cost: Optional[float] = None


class IHSSolver:
    def __init__(self, wcnf: WCNF, config: Optional[IHSConfig] = None):
        self.wcnf = wcnf
        self.config = config or IHSConfig()
        self.cores: list[frozenset[int]] = list(self.config.warm_start_cores)
        self.stats = IHSStats()
        self.n_soft = len(wcnf.soft)
        self.weights = [int(w) for w in wcnf.wght]

        # Selector encoding: for each soft clause s_i, fresh variable sel_i
        # and hard clause (s_i ∨ ¬sel_i). Asserting sel_i enforces s_i;
        # asserting ¬sel_i lets the solver falsify s_i freely.
        base_var = max(
            wcnf.nv,
            max((abs(l) for cl in wcnf.soft + wcnf.hard for l in cl), default=0),
        )
        self.selectors = [base_var + 1 + i for i in range(self.n_soft)]
        self._sel_to_idx = {s: i for i, s in enumerate(self.selectors)}

        self._sat = Solver(name=self.config.sat_name)
        for cl in wcnf.hard:
            self._sat.add_clause(cl)
        for sel, soft_cl in zip(self.selectors, wcnf.soft):
            self._sat.add_clause(list(soft_cl) + [-sel])

        # State maintained across iterations for stabilization
        self._stability_center: Optional[frozenset[int]] = None
        self._alpha = self.config.wentges_alpha0
        self._reference: Optional[frozenset[int]] = None
        self._last_fractional: Optional[list[float]] = None

        # Abstract cores: rank-k master rows (indices, rhs) with rhs >= 2,
        # plus one totalizer per abstraction set counting dropped selectors.
        self.abstract_rows: list = []
        self.abstraction_sets: list = []
        self._totalizers: list = []
        self._cnt_to_setlevel: dict = {}
        self._abstracted: frozenset = frozenset()
        self._init_abstraction(base_var + self.n_soft)

    def _init_abstraction(self, top_id: int) -> None:
        sets = self.config.abstraction_sets
        if sets is None and self.config.auto_abstraction:
            by_weight: dict = {}
            for i, w in enumerate(self.weights):
                by_weight.setdefault(w, []).append(i)
            sets = [idxs for idxs in by_weight.values() if len(idxs) >= 3]
        if not sets:
            return
        self.abstraction_sets = [frozenset(s) for s in sets]
        covered: set = set()
        for A in self.abstraction_sets:
            if len({self.weights[i] for i in A}) != 1:
                raise ValueError("abstraction sets must have uniform weight")
            if A & covered:
                raise ValueError("abstraction sets must be disjoint")
            covered |= A
        self._abstracted = frozenset(covered)

        top = top_id
        for set_idx, A in enumerate(self.abstraction_sets):
            lits = [-self.selectors[i] for i in sorted(A)]
            tot = ITotalizer(lits=lits, ubound=len(A), top_id=top)
            top = tot.top_id
            self._sat.append_formula(tot.cnf.clauses)
            self._totalizers.append(tot)
            # Assuming -tot.rhs[t] enforces "at most t clauses of A dropped".
            for t in range(len(A)):
                self._cnt_to_setlevel[-tot.rhs[t]] = (set_idx, t)

    # ------------------------------------------------------------------ master

    def _solve_master(self) -> frozenset[int]:
        """Solve the hitting-set IP master. Returns the set of indices to drop."""
        self.stats.ip_solves += 1
        prob = pulp.LpProblem("ihs_master", pulp.LpMinimize)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(self.n_soft)]

        obj_terms = [self.weights[i] * y[i] for i in range(self.n_soft)]

        # Wentges stabilization: penalize deviation from the stability center.
        # For binary y_i and binary center c_i, |y_i - c_i| = y_i if c_i=0
        # else (1 - y_i); the constants drop out of the argmin.
        if self._alpha > 0 and self._stability_center is not None:
            for i in range(self.n_soft):
                if i in self._stability_center:
                    obj_terms.append(self._alpha * (1 - y[i]))
                else:
                    obj_terms.append(self._alpha * y[i])

        prob += pulp.lpSum(obj_terms)

        for core in self.cores:
            prob += pulp.lpSum(y[i] for i in core) >= 1
        self._add_abstract_rows(prob, y, integer=True)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        H_star = frozenset(i for i in range(self.n_soft) if y[i].varValue > 0.5)

        # Solve the LP relaxation too — used for Pareto selection and price-aware
        # ordering. We do this with the unstabilized objective for clean prices.
        self._last_fractional_vals, self._last_djs, self._last_lp_obj = self._solve_master_lp()
        self.stats.lp_value = self._last_lp_obj

        return H_star

    def _add_abstract_rows(self, prob, y: list, integer: bool) -> None:
        """Encode the abstract-core disjunctions (see `_core_to_row`).

        Each row gets an indicator z per counting term, with
        Σ_{i∈A} y_i ≥ (t+1)·z and Σ z + Σ_{ordinary} y_j ≥ 1. With binary
        z this is exact (it always cuts off the H* that produced the row);
        its LP relaxation projects onto the normalized row, so the LP
        master is tightened as well. A single-set row without ordinaries
        collapses to the pure rank-k row Σ_{i∈A} y_i ≥ t+1.
        """
        for r_idx, (cnt_terms, ord_idxs) in enumerate(self.abstract_rows):
            zs = []
            for (set_idx, t) in sorted(cnt_terms):
                if integer:
                    z = pulp.LpVariable(f"z_{r_idx}_{set_idx}_{t}", cat="Binary")
                else:
                    z = pulp.LpVariable(f"z_{r_idx}_{set_idx}_{t}", lowBound=0, upBound=1)
                prob += pulp.lpSum(
                    y[i] for i in self.abstraction_sets[set_idx]
                ) >= (t + 1) * z
                zs.append(z)
            prob += pulp.lpSum(zs) + pulp.lpSum(y[j] for j in ord_idxs) >= 1

    def _solve_master_lp(self) -> tuple[list[float], list[float], float]:
        """Solve the LP relaxation and return (values, reduced_costs, objective)."""
        prob = pulp.LpProblem("ihs_master_lp", pulp.LpMinimize)
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1) for i in range(self.n_soft)]
        prob += pulp.lpSum(self.weights[i] * y[i] for i in range(self.n_soft))
        for core in self.cores:
            prob += pulp.lpSum(y[i] for i in core) >= 1
        self._add_abstract_rows(prob, y, integer=False)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        vals = [y[i].varValue or 0.0 for i in range(self.n_soft)]
        djs = [y[i].dj or 0.0 for i in range(self.n_soft)]
        obj = pulp.value(prob.objective)
        return vals, djs, obj

    # -------------------------------------------------------------- probe set

    def _probe_set(self, H_star: frozenset[int]) -> frozenset[int]:
        """Apply in-out separation: interpolate H* with a reference set."""
        if self.config.in_out_lambda <= 0 or self._reference is None:
            return H_star
        # Deterministic in-out: take the intersection if lambda >= 0.5,
        # otherwise the symmetric difference falls back to H*. This is a
        # conservative variant that probes a "more central" hitting set.
        if self.config.in_out_lambda >= 0.5:
            return H_star & self._reference
        return H_star

    # --------------------------------------------------------------- sat call

    def _build_assumptions(self, dropped: frozenset[int]) -> list[int]:
        """Selectors for enforced clauses; price-aware order if requested."""
        enforced = [i for i in range(self.n_soft) if i not in dropped]

        if self.config.price_aware_order and hasattr(self, '_last_fractional_vals'):
            # Put high-price (large dual-LP value) selectors first so the
            # SAT solver branches on them earlier, biasing cores toward them.
            enforced.sort(key=lambda i: -self._last_fractional_vals[i])

        return [self.selectors[i] for i in enforced]

    def _sat_call(self, assumptions: list[int]) -> tuple[bool, list[int]]:
        self.stats.sat_calls += 1
        sat = self._sat.solve(assumptions=assumptions)
        if sat:
            return True, []
        raw_core = self._sat.get_core() or []
        return False, raw_core

    def _core_to_indices(self, raw_core: list[int]) -> frozenset[int]:
        return frozenset(self._sel_to_idx[lit] for lit in raw_core if lit in self._sel_to_idx)

    # ------------------------------------------------------------- abstraction

    def _build_abstract_assumptions(self, dropped: frozenset[int]) -> list[int]:
        """Assumptions for the abstract probe at H*.

        Clauses outside every abstraction set are enforced individually
        (unless dropped); for each abstraction set A only the *count*
        t_A = |H* ∩ A| is enforced, via the counting literal asserting
        "at most t_A clauses of A are dropped". Because weights are uniform
        within each set, a SAT answer certifies a solution of cost at most
        cost(H*), so the standard termination argument goes through.
        """
        assumptions = [
            self.selectors[i]
            for i in range(self.n_soft)
            if i not in self._abstracted and i not in dropped
        ]
        for set_idx, A in enumerate(self.abstraction_sets):
            t = len(A & dropped)
            if t < len(A):
                assumptions.append(-self._totalizers[set_idx].rhs[t])
        return assumptions

    def _core_to_row(self, raw_core: list[int]) -> tuple[frozenset, frozenset]:
        """Translate an abstract core into a master row.

        The core is a *disjunction*: over its counting literals,
        "more than t_A clauses of A are falsified" for some A, or some
        core ordinary clause is falsified. Its valid linear surrogate is
        the normalized row

            Σ_{(A,t)} (1/(t+1)) Σ_{i∈A} y_i  +  Σ_{ordinary j} y_j  ≥  1,

        since each disjunct alone pushes the left-hand side to at least 1.
        For a single-set core without ordinaries this is exactly the
        rank-k row Σ_{i∈A} y_i ≥ t+1. Returned as (cnt_terms, ord_idxs)
        with cnt_terms a frozenset of (set_idx, t) pairs.
        """
        cnt_terms: set = set()
        ord_idxs: set = set()
        for lit in raw_core:
            if lit in self._sel_to_idx:
                ord_idxs.add(self._sel_to_idx[lit])
            elif lit in self._cnt_to_setlevel:
                cnt_terms.add(self._cnt_to_setlevel[lit])
        return frozenset(cnt_terms), frozenset(ord_idxs)

    # ------------------------------------------------------------------ sweep

    def _threshold_sweep(self, known: set) -> Optional[frozenset[int]]:
        """Price-guided separation against the fractional prices ȳ.

        Binary-search the least price level θ such that enforcing
        S(θ) = {s_i : ȳ_i ≤ θ} is UNSAT, and return the core found there
        (supported entirely on the cheapest-priced clauses that support any
        core). Heuristic separator: it may return a known core, in which
        case the caller falls through to the standard probe at H*.
        """
        if not hasattr(self, '_last_fractional_vals'):
            return None
        y_bar = self._last_fractional_vals
        levels = sorted(set(y_bar))
        lo, hi = 0, len(levels) - 1
        best_core = None
        while lo <= hi:
            mid = (lo + hi) // 2
            dropped = frozenset(
                i for i in range(self.n_soft) if y_bar[i] > levels[mid] + 1e-9
            )
            self.stats.sweep_calls += 1
            sat, raw_core = self._sat_call(self._build_assumptions(dropped))
            if sat:
                lo = mid + 1
            else:
                core = self._core_to_indices(raw_core)
                if core:
                    best_core = core
                hi = mid - 1
        if best_core is not None and best_core not in known:
            return best_core
        return None

    # --------------------------------------------------------------- pareto

    def _pareto_refine(self, base_core: frozenset[int], dropped: frozenset[int], known: set[frozenset[int]]) -> frozenset[int]:
        """Search for a core with larger violation than `base_core` against ȳ.

        Strategy: among the currently enforced selectors, additionally enforce
        the unhit (low-ȳ) ones one at a time and re-call SAT. If a deeper
        core comes back, prefer it. Bounded by a small fan-out for tractability.
        """
        if not hasattr(self, '_last_fractional_vals'):
            return base_core
        y_bar = self._last_fractional_vals
        best_core = base_core
        best_violation = 1.0 - sum(y_bar[i] for i in base_core)

        # Candidates: indices NOT in current core, ordered by ascending y_bar
        # (i.e., "least likely to be hit" first — most useful to add).
        candidates = sorted(
            (i for i in range(self.n_soft) if i not in base_core and i in dropped),
            key=lambda i: y_bar[i],
        )[:5]

        for cand in candidates:
            trial_dropped = dropped - {cand}
            assumptions = self._build_assumptions(trial_dropped)
            sat, raw_core = self._sat_call(assumptions)
            if sat:
                continue
            trial_core = self._core_to_indices(raw_core)
            if trial_core in known:
                continue
            violation = 1.0 - sum(y_bar[i] for i in trial_core)
            if violation > best_violation + 1e-9:
                best_core = trial_core
                best_violation = violation
                self.stats.pareto_refinements += 1

        return best_core

    # ----------------------------------------------------------------- solve

    def _get_upper_bound(self) -> float:
        """Get a heuristic upper bound by solving the hard clauses."""
        sat = self._sat.solve()
        if not sat:
            return float('inf')
        model = self._sat.get_model()
        cost = 0.0
        # Check which soft clauses are falsified by this model
        for i, soft_cl in enumerate(self.wcnf.soft):
            satisfied = False
            for lit in soft_cl:
                if (lit > 0 and model[abs(lit)-1] > 0) or (lit < 0 and model[abs(lit)-1] < 0):
                    satisfied = True
                    break
            if not satisfied:
                cost += self.weights[i]
        return cost

    def solve(self) -> tuple[float, frozenset[int]]:
        known = set(self.cores)
        known_rows = set(self.abstract_rows)
        fixed_satisfied = set()
        ub = self._get_upper_bound()

        for _ in range(self.config.max_iters):
            self.stats.iterations += 1

            H_star = self._solve_master()

            if self.config.reduced_cost_fixing and hasattr(self, '_last_lp_obj'):
                # Reduced cost fixing: if lb + red_cost > ub, then y_i must be 0 (clause satisfied)
                for i in range(self.n_soft):
                    if i in fixed_satisfied:
                        continue
                    if self._last_lp_obj + self._last_djs[i] > ub + 1e-6:
                        self._sat.add_clause(list(self.wcnf.soft[i]))
                        fixed_satisfied.add(i)

            # Price-guided threshold sweep: a bound-improvement phase between
            # master solves. If it separates a fresh core against the
            # fractional prices, add it and re-solve the master; otherwise
            # fall through to the terminal probe at H*.
            if self.config.threshold_sweep:
                sweep_core = self._threshold_sweep(known)
                if sweep_core is not None:
                    self.cores.append(sweep_core)
                    known.add(sweep_core)
                    self.stats.cores_found += 1
                    continue

            # Abstract-core probe: enforce per-set falsification *counts*
            # instead of the specific clauses of H*. UNSAT yields a rank-k
            # row; SAT certifies optimality (uniform weights within sets).
            if self.abstraction_sets:
                assumptions = self._build_abstract_assumptions(H_star)
                sat, raw_core = self._sat_call(assumptions)
                if sat:
                    cost = sum(self.weights[i] for i in H_star)
                    self.stats.cost = cost
                    return cost, H_star
                cnt_terms, ord_idxs = self._core_to_row(raw_core)
                if not cnt_terms and not ord_idxs:
                    break
                if not cnt_terms:
                    # Purely ordinary core: a plain hitting-set row.
                    if ord_idxs not in known:
                        self.cores.append(ord_idxs)
                        known.add(ord_idxs)
                        self.stats.cores_found += 1
                else:
                    row = (cnt_terms, ord_idxs)
                    if row not in known_rows:
                        self.abstract_rows.append(row)
                        known_rows.add(row)
                        self.stats.abstract_rows += 1
                    # A core mixing several counting sets yields only a weak
                    # disjunctive row (its LP projection is the normalized
                    # surrogate). Supplement it with an ordinary core from
                    # the same H*, as Berg-Bacchus keep ordinary cores
                    # alongside abstract ones. The plain probe is UNSAT
                    # whenever the abstract probe is (it is more constrained).
                    if len(cnt_terms) >= 2:
                        _, raw_core = self._sat_call(self._build_assumptions(H_star))
                        ord_core = self._core_to_indices(raw_core)
                        if ord_core and ord_core not in known:
                            self.cores.append(ord_core)
                            known.add(ord_core)
                            self.stats.cores_found += 1
                continue

            probe_dropped = self._probe_set(H_star)

            # 1. Probe at H_star (or in-out point) — always done.
            assumptions = self._build_assumptions(probe_dropped)
            sat, raw_core = self._sat_call(assumptions)

            if sat:
                cost = sum(self.weights[i] for i in H_star)
                self.stats.cost = cost
                return cost, H_star

            core = self._core_to_indices(raw_core)
            
            if core:
                if self.config.pareto_selection:
                    core = self._pareto_refine(core, probe_dropped, known)
                
                if core not in known:
                    self.cores.append(core)
                    known.add(core)
                    self.stats.cores_found += 1
                
                # Multi-core extraction: find disjoint cores
                if self.config.multi_core:
                    current_dropped = set(probe_dropped) | core
                    while True:
                        assumptions = self._build_assumptions(frozenset(current_dropped))
                        sat, raw_core = self._sat_call(assumptions)
                        if sat:
                            break
                        next_core = self._core_to_indices(raw_core)
                        if not next_core or next_core in known:
                            break
                        self.cores.append(next_core)
                        known.add(next_core)
                        self.stats.cores_found += 1
                        current_dropped |= next_core

            # If the in-out probe returned a duplicate (stagnation), fall back
            # to probing at H_star directly to make sure we generate a fresh
            # cut against the master's actual choice.
            if (probe_dropped != H_star) and (not core or core in known):
                sat, raw_core = self._sat_call(self._build_assumptions(H_star))
                if sat:
                    cost = sum(self.weights[i] for i in H_star)
                    self.stats.cost = cost
                    return cost, H_star
                core = self._core_to_indices(raw_core)
                if core and core not in known:
                    self.cores.append(core)
                    known.add(core)
                    self.stats.cores_found += 1

            if not core and not self.config.multi_core:
                break

            # Update stabilization state.
            if self.config.wentges_alpha0 > 0:
                self._stability_center = H_star
                self._alpha *= self.config.wentges_decay
            if self._reference is None:
                self._reference = H_star

        raise RuntimeError(f"IHS did not converge within {self.config.max_iters} iterations")

    def close(self):
        self._sat.delete()
