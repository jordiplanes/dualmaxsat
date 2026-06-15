"""Load WCNF instances from disk and run each §7 technique on each."""

from pathlib import Path

from pysat.formula import WCNF

from coreguided import harvest_cores
from ihs import IHSConfig, IHSSolver

WCNF_DIR = Path(__file__).parent / "wcnf"


def load(name: str) -> WCNF:
    return WCNF(from_file=str(WCNF_DIR / name))


def run(label: str, wcnf_path: str, config: IHSConfig) -> None:
    solver = IHSSolver(load(wcnf_path), config)
    try:
        cost, _ = solver.solve()
    finally:
        solver.close()
    s = solver.stats
    lp_val = sum(config.weights[i] * y_lp for i, y_lp in enumerate(solver._last_fractional)) if solver._last_fractional else "N/A"
    print(
        f"  {label:34s}  cost={cost:>3}  lp_val={lp_val:>4}  iter={s.iterations:>2}  "
        f"new_cores={s.cores_found:>2}  sat={s.sat_calls:>2}  "
        f"ip={s.ip_solves:>2}  pareto_refs={s.pareto_refinements}"
    )


def compare(title: str, wcnf_path: str) -> None:
    print()
    print("=" * 100)
    print(f"{title}  [{wcnf_path}]")
    print("=" * 100)

    wcnf = load(wcnf_path)
    seed = harvest_cores(wcnf, budget=5)
    print(f"  (warm-start seed harvested {len(seed)} cores)")

    # Pass weights to the print statement
    class ConfigWithWeights(IHSConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.weights = [int(w) for w in wcnf.wght]

    def run_local(label, cfg):
        solver = IHSSolver(wcnf, cfg)
        try:
            cost, _ = solver.solve()
        finally:
            solver.close()
        s = solver.stats
        lp_val = solver._last_lp_obj if hasattr(solver, '_last_lp_obj') else 0.0
        print(
            f"  {label:34s}  cost={cost:>4}  lp_val={lp_val:>4.1f}  iter={s.iterations:>2}  "
            f"new_cores={s.cores_found:>2}  sat={s.sat_calls:>2}  "
            f"ip={s.ip_solves:>2}  pareto_refs={s.pareto_refinements}"
        )

    run_local("Plain IHS (baseline)",        IHSConfig())
    run_local("+ warm-start (Fu-Malik seed)", IHSConfig(warm_start_cores=seed))
    run_local("+ Pareto core selection",     IHSConfig(pareto_selection=True))
    run_local("+ Wentges (α₀=1.0, ρ=0.5)",   IHSConfig(wentges_alpha0=1.0, wentges_decay=0.5))
    run_local("+ in-out separation (λ=0.5)",  IHSConfig(in_out_lambda=0.5))
    run_local("+ price-aware ordering",      IHSConfig(price_aware_order=True))
    run_local("+ Multi-core extraction",     IHSConfig(multi_core=True))
    run_local("+ Reduced cost fixing",       IHSConfig(reduced_cost_fixing=True))
    run_local(
        "+ HYBRID (all techniques)",
        IHSConfig(
            warm_start_cores=seed,
            pareto_selection=True,
            wentges_alpha0=1.0,
            wentges_decay=0.5,
            in_out_lambda=0.5,
            price_aware_order=True,
            multi_core=True,
            reduced_cost_fixing=True,
        ),
    )


def main():
    compare("Worked example (§5): 4 unit soft clauses, optimum 2", "worked.wcnf")
    compare("Triangle counterexample (§6): K3, LP=1.5, IP=2", "k3.wcnf")
    compare("Weighted chain: 10 soft clauses + hard tie", "weighted_chain.wcnf")


if __name__ == "__main__":
    main()
