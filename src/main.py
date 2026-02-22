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
        "best_f": best_f,
        "error": error,
        "generations": optimizer.generation,
        "history": optimizer.history,
        "best_x": best_x
    }

def run_multiple_experiments(func_name, n, runs=10, sigma0=0.3):
    results = []
    for i in range(runs):
        r = run_experiment(func_name, n, sigma0, seed=i, verbose=False)
        results.append(r)
        print(f"  [{func_name}] Run {i+1}/{runs} | best_f = {r['best_f']:.4e} | gen = {r['generations']}")

    best_vals = [r["best_f"] for r in results]
    print(f"\n  Statistika ({runs} pokretanja):")
    print(f"  Srednja vrednost : {np.mean(best_vals):.4e}")
    print(f"  Standardna devijacija : {np.std(best_vals):.4e}")
    print(f"  Minimum : {np.min(best_vals):.4e}")
    print(f"  Maksimum : {np.max(best_vals):.4e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CMA-ES Benchmark Runner")
    parser.add_argument("--func",  type=str, default="all",
                        help="Naziv funkcije (ili 'all'): " + ", ".join(BENCHMARKS.keys()))
    parser.add_argument("--n",     type=int, default=10,   help="Dimenzija prostora (default: 10)")
    parser.add_argument("--sigma", type=float, default=0.3, help="Inicijalni korak sigma (default: 0.3)")
    parser.add_argument("--runs",  type=int, default=1,    help="Broj nezavisnih pokretanja (default: 1)")
    parser.add_argument("--seed",  type=int, default=42,   help="Random seed (default: 42)")
    args = parser.parse_args()

    funcs = list(BENCHMARKS.keys()) if args.func == "all" else [args.func]

    print("=" * 60)
    print(f"CMA-ES eksperiment | n={args.n} | sigma0={args.sigma}")
    print("=" * 60)

    all_results = {}
    for fname in funcs:
        print(f"\n Funkcija: {fname}")
        print("-" * 40)
        if args.runs == 1:
            r = run_experiment(fname, args.n, sigma0=args.sigma, seed=args.seed)
            all_results[fname] = [r]
        else:
            all_results[fname] = run_multiple_experiments(fname, args.n, runs=args.runs, sigma0=args.sigma)

    print("\n" + "=" * 60)
    print("SAŽETAK REZULTATA")
    print("=" * 60)
    print(f"{'Funkcija':<14} {'best_f':>14} {'Greška':>12} {'Generacije':>12}")
    print("-" * 55)
    for fname, results in all_results.items():
        best_f_vals = [r["best_f"] for r in results]
        gen_vals    = [r["generations"] for r in results]
        err_vals    = [r["error"] for r in results]
        mean_f_vals = np.mean(best_f_vals)
        print(f"{fname:<14} {np.mean(best_f_vals):>14.4e} {np.mean(err_vals):>12.4e} {int(np.mean(gen_vals)):>12d}")


if __name__ == "__main__":
    main()