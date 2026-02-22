import numpy as np 

def sphere(x):
    """
    Sphere funkcija: f(x) = Σ xᵢ²
    Minimum: f(0) = 0
    Karakteristike: konveksna, separabilna, izotropna - najlakša za optimizaciju
    """
    return np.sum(x ** 2)

def ellipsoid(x):
    """
    Ellipsoid funkcija: f(x) = Σ (1000^(i/(n-1)) * xᵢ)²
    Minimum: f(0) = 0
    Karakteristike: jako izdužena, testira adaptaciju kovarijansne matrice
    """
    n = len(x)
    exponents = np.arange(n) / (n - 1) if n > 1 else np.array([0.0])
    return np.sum((1000 ** exponents * x) ** 2)

def rastrigin(x):
    """
    Rastrigin funkcija: f(x) = 10n + Σ [xᵢ² - 10·cos(2πxᵢ)]
    Minimum: f(0) = 0
    Karakteristike: jako multimodalna, mnogo lokalnih minimuma - teška
    """
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """
    Rosenbrock funkcija: f(x) = Σ [100(x_{i+1} - xᵢ²)² + (1 - xᵢ)²]
    Minimum: f(1, 1, ..., 1) = 0
    Karakteristike: uska zakrivljena dolina, teška za gradijentne metode
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley(x):
    """
    Ackley funkcija: f(x) = -20·exp(-0.2·√(1/n · Σxᵢ²)) - exp(1/n · Σcos(2πxᵢ)) + 20 + e    Minimum: f(0) = 0
    Karakteristike: mnogo lokalnih minimuma,  ali jedan jasno globalni minimum, ravna spoljašnja regija i strm centralni bazen,  - teska 
    """
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq  = np.sum(x ** 2) / n
    sum_cos = np.sum(np.cos(c * x)) / n
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e


BENCHMARKS = {
    "Sphere":     {"f": sphere,     "x_opt": lambda n: np.zeros(n), "domain": (-5, 5), "tol": 1e-6},
    "Ellipsoid":  {"f": ellipsoid,  "x_opt": lambda n: np.zeros(n), "domain": (-5, 5), "tol": 1e-6},
    "Rastrigin":  {"f": rastrigin,  "x_opt": lambda n: np.zeros(n), "domain": (-5.12, 5.12), "tol": 1e-4},
    "Rosenbrock": {"f": rosenbrock, "x_opt": lambda n: np.ones(n),  "domain": (-2, 2), "tol": 1e-4},
    "Ackley":     {"f": ackley,     "x_opt": lambda n: np.zeros(n), "domain": (-32, 32),  "tol": 1e-4},
}


"""Vraća benchmark funkciju po imenu."""    
def get_function(name):
    if name not in BENCHMARKS:
        raise ValueError(f"Nepoznata funkcija '{name}'. Dostupne: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]