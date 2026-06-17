import time
from benchmarks import generate_random_max3sat
from ihs import IHSConfig, IHSSolver

def test():
    for v in [30, 40, 50, 60, 70, 80]:
        c = v * 5
        wcnf = generate_random_max3sat(v, c, seed=42, weighted=True)
        solver = IHSSolver(wcnf, IHSConfig())
        start = time.time()
        try:
            solver.solve()
            t = time.time() - start
            print(f"V={v}: {t:.2f}s")
        except Exception as e:
            print(f"V={v}: FAILED ({e})")

if __name__ == "__main__":
    test()
