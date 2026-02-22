# Usage: python3 src/main.py --func = [naziv_funkcije]

import argparse
import numpy as np
from cmaes import CMAES
from benchmark import BENCHMARKS, get_function

def run_experiment(func_name, n, sigma0 = 0.3, seed=None, verbose=True):
    if seed is not None:
        np.random.seed(seed)

    bench = get_function(func_name)
    f     = bench["f"]
    lo, hi = bench["domain"]    

    m0 = np.random.uniform(lo * 0.8, hi * 0.8, size=n)

    optimizer = CMAES(f=f, n=n, m0=m0, sigma0=sigma0)
    best_x, best_f = optimizer.run(verbose=verbose)

    x_opt = bench["x_opt"](n)
    error = np.linalg.norm(best_x - x_opt)

    return {
        "func": func_name,
        "n": n,
        "best_f": f,
        "error": error,
        "generations": optimizer.generation,
        "history": optimizer.history,
        "best_x": best_x
    }

def main():
    parser = argparse.ArgumentParser(description="CMA-ES Benchmark Runner")
    parser.add_argument("--func",  type=str, default="Sphere",
                        help="Naziv funkcije : " + ", ".join(BENCHMARKS.keys()))
    
    args = parser.parse_args()  
    func_name = args.func

    r = run_experiment(func_name, 10, 0.3, None, verbose=False)
    for name, value in r.items():
        print(name, value)

if __name__ == "__main__":
    main()