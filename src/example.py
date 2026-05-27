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
    print(
        f"  {label:34s}  cost={cost:>3}  iter={s.iterations:>2}  "
        f"new_cores={s.cores_found:>2}  sat={s.sat_calls:>2}  "
        f"ip={s.ip_solves:>2}  pareto_refs={s.pareto_refinements}"
    )


def compare(title: str, wcnf_path: str) -> None:
    print()
    print("=" * 100)
    print(f"{title}  [{wcnf_path}]")
    print("=" * 100)

    seed = harvest_cores(load(wcnf_path), budget=5)
    print(f"  (warm-start seed harvested {len(seed)} cores)")

    run("Plain IHS (baseline)",        wcnf_path, IHSConfig())
    run("+ warm-start (Fu-Malik seed)", wcnf_path, IHSConfig(warm_start_cores=seed))
    run("+ Pareto core selection",     wcnf_path, IHSConfig(pareto_selection=True))
    run("+ Wentges (α₀=1.0, ρ=0.5)",   wcnf_path, IHSConfig(wentges_alpha0=1.0, wentges_decay=0.5))
    run("+ in-out separation (λ=0.5)",  wcnf_path, IHSConfig(in_out_lambda=0.5))
    run("+ price-aware ordering",      wcnf_path, IHSConfig(price_aware_order=True))
    run(
        "+ HYBRID (all techniques)",
        wcnf_path,
        IHSConfig(
            warm_start_cores=seed,
            pareto_selection=True,
            wentges_alpha0=1.0,
            wentges_decay=0.5,
            in_out_lambda=0.5,
            price_aware_order=True,
        ),
    )


def main():
    compare("Worked example (§5): 4 unit soft clauses, optimum 2", "worked.wcnf")
    compare("Weighted chain: 10 soft clauses + hard tie", "weighted_chain.wcnf")


if __name__ == "__main__":
    main()
