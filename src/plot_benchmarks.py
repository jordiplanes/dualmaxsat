import matplotlib.pyplot as plt
import time
import os
from benchmarks import generate_vertex_cover
from example import harvest_cores
from ihs import IHSConfig, IHSSolver

def run_solver(wcnf, config, label, timeout=20):
    solver = IHSSolver(wcnf, config)
    start_time = time.time()
    try:
        # DualMaxSAT solver doesn't natively support a timeout parameter in solve()
        # but we can just let it run. We'll rely on our size scaling to keep it ~10s.
        cost, _ = solver.solve()
    finally:
        solver.close()
    end_time = time.time()
    return solver.stats.iterations, end_time - start_time

def main():
    sizes = [20, 30, 40, 47]
    p_edge = 0.2
    
    config_labels = [
        "Baseline",
        "+ Warm-start",
        "+ Pareto",
        "+ Stabilization",
        "+ Multi-core",
        "+ RC Fixing",
        "Hybrid (All)"
    ]
    
    # Dictionaries to store results: dict[label] = list of values
    iters_data = {label: [] for label in config_labels}
    times_data = {label: [] for label in config_labels}
    
    print("Running experiments for plotting (up to ~10s per baseline)...")
    
    for size in sizes:
        print(f"\\nTesting Vertex Cover on G(n={size}, p={p_edge})")
        wcnf, edges = generate_vertex_cover(size, p_edge, seed=42)
        
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
            iters_data[label].append(it)
            times_data[label].append(t)
            print(f"  {label:18s}: {it:3d} iters, {t:5.2f}s")
            
    # Plot Iterations
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for label in config_labels:
        # Use different markers for distinction
        marker = 'o' if label == "Baseline" else ('s' if label == "Hybrid (All)" else '^')
        linestyle = '-' if label in ["Baseline", "Hybrid (All)"] else '--'
        plt.plot(sizes, iters_data[label], marker=marker, linestyle=linestyle, label=label)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Solver Iterations')
    plt.title('Iterations vs. Graph Size')
    plt.legend()
    plt.grid(True)
    
    # Plot Time
    plt.subplot(1, 2, 2)
    for label in config_labels:
        marker = 'o' if label == "Baseline" else ('s' if label == "Hybrid (All)" else '^')
        linestyle = '-' if label in ["Baseline", "Hybrid (All)"] else '--'
        plt.plot(sizes, times_data[label], marker=marker, linestyle=linestyle, label=label)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (Seconds)')
    plt.title('Time vs. Graph Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save for repo
    repo_path = "/Users/jordiplanes/repos/dualmaxsat/benchmark_plot.png"
    plt.savefig(repo_path)
    
    # Save for artifact
    artifact_path = "/Users/jordiplanes/.gemini/antigravity-cli/brain/6eff1fcf-eac1-4881-9e18-c1c1de376340/benchmark_plot.png"
    plt.savefig(artifact_path)
    print(f"\\nSaved plot to {repo_path} and {artifact_path}")

if __name__ == "__main__":
    main()
