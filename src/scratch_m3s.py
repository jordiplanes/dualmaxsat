import time
from benchmarks import generate_random_max3sat
from ihs import IHSConfig, IHSSolver

def test():
    for v in range(16, 25):
        c = v * 6
        wcnf = generate_random_max3sat(v, c, seed=42, weighted=True)
        solver = IHSSolver(wcnf, IHSConfig(max_iters=5000))
        start = time.time()
        try:
            solver.solve()
            t = time.time() - start
            print(f"V={v}, C={c}: {t:.2f}s, {solver.stats.iterations} iters")
        except Exception as e:
            print(f"V={v}, C={c}: FAILED ({e})")

if __name__ == "__main__":
    test()
