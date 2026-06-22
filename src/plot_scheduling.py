import matplotlib.pyplot as plt
import time
from scheduling_benchmarks import generate_task_allocation, generate_gate_allocation, generate_shift_scheduling
from example import harvest_cores
from ihs import IHSConfig, IHSSolver

def run_solver(wcnf, config):
    solver = IHSSolver(wcnf, config)
    start_time = time.time()
    try:
        solver.solve()
    finally:
        solver.close()
    return solver.stats.iterations, time.time() - start_time

config_labels = [
    "Baseline",
    "+ Multi-core",
    "Hybrid (All)"
]

def get_configs(wcnf):
    seed_cores = harvest_cores(wcnf, budget=5)
    return {
        "Baseline": IHSConfig(),
        "+ Multi-core": IHSConfig(multi_core=True),
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

def plot_experiment(title, sizes, xlabel, generator, *gen_args, filename="plot.png", seed=123):
    iters_data = {label: [] for label in config_labels}
    times_data = {label: [] for label in config_labels}
    
    print(f"\nRunning {title}...")
    for size in sizes:
        wcnf = generator(size, *gen_args, seed=seed)
        configs = get_configs(wcnf)
        
        for label in config_labels:
            it, t = run_solver(wcnf, configs[label])
            iters_data[label].append(it)
            times_data[label].append(t)
            print(f"  {title} (size={size}) {label}: {it} iters, {t:.2f}s")
            
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for label in config_labels:
        plt.plot(sizes, iters_data[label], marker='o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel('Solver Iterations')
    plt.title(f'{title} - Iterations')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for label in config_labels:
        plt.plot(sizes, times_data[label], marker='o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel('Time (s)')
    plt.title(f'{title} - Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    repo_path = f"/Users/jordiplanes/repos/dualmaxsat/{filename}"
    plt.savefig(repo_path)
    plt.close()
    print(f"Saved {filename}")

def wrap_shift_scheduling(n_days, seed=42):
    # Fix n_emps=4, req_per_day=2. Vary n_days.
    return generate_shift_scheduling(4, n_days, 2, seed=seed)

def main():
    # 1. Task Allocation
    # Extended sizes to push beyond 10s execution
    plot_experiment("Task Allocation", [6, 8, 10, 12, 14], "Number of Tasks", 
                    generate_task_allocation, 3, 5, seed=123, filename="scheduling_plot_1.png")
    
    # 2. Gate Allocation
    plot_experiment("Gate Allocation", [5, 6, 7, 8], "Number of Flights", 
                    generate_gate_allocation, 3, 0.3, seed=123, filename="scheduling_plot_2.png")
    
    # 3. Shift Scheduling
    plot_experiment("Shift Scheduling", [4, 5, 6, 7], "Number of Days", 
                    wrap_shift_scheduling, seed=123, filename="scheduling_plot_3.png")

if __name__ == "__main__":
    main()
