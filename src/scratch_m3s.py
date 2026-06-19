import time
from benchmarks import generate_random_max3sat
from ihs import IHSConfig, IHSSolver

def test():
    for v in [40, 42, 44, 46, 48, 50]:
        c = v * 5
        wcnf = generate_random_max3sat(v, c, seed=42, weighted=True)
        solver = IHSSolver(wcnf, IHSConfig(max_iters=10000))
        start = time.time()
        try:
            solver.solve()
            t = time.time() - start
            print(f"V={v}: {t:.2f}s, {solver.stats.iterations} iters")
        except Exception as e:
            print(f"V={v}: FAILED ({e})")

if __name__ == "__main__":
    test()
