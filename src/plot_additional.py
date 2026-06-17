import matplotlib.pyplot as plt
import time
from benchmarks import generate_set_cover, generate_random_max3sat
from example import harvest_cores
from ihs import IHSConfig, IHSSolver

def run_solver(wcnf, config, label, timeout=30):
    solver = IHSSolver(wcnf, config)
    start_time = time.time()
    try:
        cost, _ = solver.solve()
    finally:
        solver.close()
    end_time = time.time()
    return solver.stats.iterations, end_time - start_time

def main():
    sc_sizes = [100, 150, 200, 300]
    m3s_sizes = [15, 17, 19, 21]
    
    config_labels = [
        "Baseline",
        "+ Warm-start",
        "+ Pareto",
        "+ Stabilization",
        "+ Multi-core",
        "+ RC Fixing",
        "Hybrid (All)"
    ]
    
    # Data structures
    sc_iters = {label: [] for label in config_labels}
    sc_times = {label: [] for label in config_labels}
    
    m3s_iters = {label: [] for label in config_labels}
    m3s_times = {label: [] for label in config_labels}
    
    print("Running Weighted Set Cover...")
    for size in sc_sizes:
        n_sets = size
        n_elements = int(size * 1.5)
        p_contain = 0.2
        print(f"\\nSet Cover (Sets={n_sets}, Elements={n_elements})")
        wcnf = generate_set_cover(n_elements, n_sets, p_contain, seed=42, weighted=True)
        seed_cores = harvest_cores(wcnf, budget=5)
        
        configs = {
            "Baseline": IHSConfig(),
            "+ Warm-start": IHSConfig(warm_start_cores=seed_cores),
            "+ Pareto": IHSConfig(pareto_selection=True),
            "+ Stabilization": IHSConfig(wentges_alpha0=1.0, wentges_decay=0.5, in_out_lambda=0.5),
            "+ Multi-core": IHSConfig(multi_core=True),
            "+ RC Fixing": IHSConfig(reduced_cost_fixing=True),
            "Hybrid (All)": IHSConfig(
                warm_start_cores=seed_cores,
                pareto_selection=True,
                wentges_alpha0=1.0,
                wentges_decay=0.5,
                in_out_lambda=0.5,
                price_aware_order=True,
                multi_core=True,
                reduced_cost_fixing=True,
            )
        }
        
        for label in config_labels:
            it, t = run_solver(wcnf, configs[label], label)
            sc_iters[label].append(it)
            sc_times[label].append(t)
            print(f"  {label:18s}: {it:3d} iters, {t:5.2f}s")
            
    print("\\nRunning Weighted Max-3SAT...")
    for size in m3s_sizes:
        n_vars = size
        n_clauses = size * 6
        print(f"\\nMax-3SAT (Vars={n_vars}, Clauses={n_clauses})")
        wcnf = generate_random_max3sat(n_vars, n_clauses, seed=42, weighted=True)
        seed_cores = harvest_cores(wcnf, budget=5)
        
        configs = {
            "Baseline": IHSConfig(),
            "+ Warm-start": IHSConfig(warm_start_cores=seed_cores),
            "+ Pareto": IHSConfig(pareto_selection=True),
            "+ Stabilization": IHSConfig(wentges_alpha0=1.0, wentges_decay=0.5, in_out_lambda=0.5),
            "+ Multi-core": IHSConfig(multi_core=True),
            "+ RC Fixing": IHSConfig(reduced_cost_fixing=True),
            "Hybrid (All)": IHSConfig(
                warm_start_cores=seed_cores,
                pareto_selection=True,
                wentges_alpha0=1.0,
                wentges_decay=0.5,
                in_out_lambda=0.5,
                price_aware_order=True,
                multi_core=True,
                reduced_cost_fixing=True,
            )
        }
        
        for label in config_labels:
            it, t = run_solver(wcnf, configs[label], label)
            m3s_iters[label].append(it)
            m3s_times[label].append(t)
            print(f"  {label:18s}: {it:3d} iters, {t:5.2f}s")
            
    # Plotting
    plt.figure(figsize=(14, 10))
    
    def plot_subplot(idx, x_sizes, data, title, ylabel):
        plt.subplot(2, 2, idx)
        for label in config_labels:
            marker = 'o' if label == "Baseline" else ('s' if label == "Hybrid (All)" else '^')
            linestyle = '-' if label in ["Baseline", "Hybrid (All)"] else '--'
            plt.plot(x_sizes, data[label], marker=marker, linestyle=linestyle, label=label)
        plt.xlabel('Size Parameter (N)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if idx == 1:
            plt.legend()
            
    plot_subplot(1, sc_sizes, sc_iters, 'Weighted Set Cover: Iterations', 'Iterations')
    plot_subplot(2, sc_sizes, sc_times, 'Weighted Set Cover: Time', 'Seconds')
    plot_subplot(3, m3s_sizes, m3s_iters, 'Weighted Max-3SAT: Iterations', 'Iterations')
    plot_subplot(4, m3s_sizes, m3s_times, 'Weighted Max-3SAT: Time', 'Seconds')
    
    plt.tight_layout()
    
    repo_path = "/Users/jordiplanes/repos/dualmaxsat/additional_plot.png"
    plt.savefig(repo_path)
    
    artifact_path = "/Users/jordiplanes/.gemini/antigravity-cli/brain/6eff1fcf-eac1-4881-9e18-c1c1de376340/additional_plot.png"
    plt.savefig(artifact_path)
    print(f"\\nSaved plot to {repo_path} and {artifact_path}")

if __name__ == "__main__":
    main()
