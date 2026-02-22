import numpy as np

# Minimizuje funkciju f bez upotrebe gradijenata
# Bazirano na Hansenovom tutorialu
class CMAES:
    
    """
        Parametri:
            f        : funkcija cilja koju minimizujemo
            n        : dimenzija prostora pretrage
            m0       : inicijalna srednja vrednost (numpy array, shape (n,))
            sigma0   : inicijalni globalni korak
            max_iter : maksimalan broj generacija
            tol      : tolerancija za kriterijum zaustavljanja
    """    
    # nzm zasto 0.3 
    def __init__(self, f, n, m0, sigma0=0.3, max_iter = 1000, tol=1e-10):
        self.f = f
        self.n = n
        self.sigma0 = sigma0
        self.max_iter = max_iter
        self.tol = tol

        # - Velicine populacije
        self.lam = 4 + int(np.floor(3 * np.log(n))) # lambda: velicina populacije
        self.mu = self.lam // 2                     # mu - broj roditelja

        # - Tezine za rekombinaciju
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = raw_w / raw_w.sum() # - normalizovane tezine
        self.mu_eff = 1.0 / (self.w ** 2).sum()

        # Parametri za adaptaciju step-size (CSA)
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.c_sigma
        # Očekivana norma N(0, I) u n dimenzija
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2)) # 

        # Parametri za adaptaciju kovarijansne matrice (CMA)
        self.c_c  = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.c_1  = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + self.mu_eff)
        )

        # Stanje algoritma
        self.m      = np.array(m0, dtype=float)         # srednja vrednost
        self.sigma  = float(sigma0)                     # globalni korak
        self.C      = np.eye(n)                         # kovarijansna matrica
        self.p_c    = np.zeros(n)                       # evolution path za C
        self.p_sigma = np.zeros(n)                      # evolution path za sigma

        # Istorija za analizu
        self.history = {
            "best_fitness": [],
            "mean_fitness": [],
            "sigma":        [],
            "best_x":       [],
        }

        self.generation = 0
        self.best_x     = None
        self.best_f     = np.inf

    """Eigen dekompozicija C = B D^2 B^T za efikasno uzorkovanje."""
    def _decompose(self):
        C_sym = (self.C + self.C.T) / 2
        eigenvalues, B = np.linalg.eigh(C_sym)
        eigenvalues = np.maximum(eigenvalues, 1e-20)    # da ne budu <= 0
        D = np.sqrt(eigenvalues)
        return B, D
    
    """
    Generiši λ kandidata:
        x_i = m + sigma * B @ diag(D) @ z_i,   z_i ~ N(0, I)
    """
    def _sample_population(self, B, D):
        z = np.random.randn(self.lam, self.n)          
        y = z * D[np.newaxis, :]                        # skaliranje po D
        y = (B @ y.T).T                                 # rotacija po B
        x = self.m + self.sigma * y                     # pomeranje po m
        return x, y, z
    
    """Ažuriranje srednje vrednosti: m' = Σ w_i * x_(i)"""      
    def _update_m(self, x_sorted):
        m_old = self.m.copy()
        self.m = self.w @ x_sorted[:self.mu]
        delta_m = self.m - m_old
        return delta_m
    
    """
    Cumulative Step-size Adaptation (CSA):
        p_sigma' = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * C^{-1/2} * delta_m/sigma
        sigma'   = sigma * exp(c_sigma/d_sigma * (||p_sigma||/chi_n - 1))
    """
    def _update_sigma(self, delta_m, B, D):
        C_invsqrt_step = B @ (np.diag(1.0 / D) @ (B.T @ (delta_m / self.sigma)))

        coef = np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + coef * C_invsqrt_step

        norm_p_sigma = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_n - 1))

    """
        Covariance Matrix Adaptation (CMA):
        Rank-1 update : koristi evolution path p_c
        Rank-μ update : koristi μ najboljih vektora iz generacije
    """    
    def _update_C(self, delta_m, y_sorted):
        # Indikator za h_sigma (gašenje rank-1 ako sigma raste prebrzo)
        n = self.n
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2 / (n + 1)) * self.chi_n
        h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1))) < threshold else 0.0

        # Azuriranje evolution path p_c 
        coef = np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * coef * (delta_m / self.sigma)

        # Rank-1 update 
        C_rank1 = np.outer(self.p_c, self.p_c)

        # Rank-μ update 
        y_mu = y_sorted[:self.mu]
        C_rankmu = sum(self.w[i] * np.outer(y_mu[i], y_mu[i]) for i in range(self.mu))

        # Kombinovano ažuriranje 
        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        self.C = ((1 - self.c_1 - self.c_mu) * self.C
                  + self.c_1 * (C_rank1 + delta_h * self.C)
                  + self.c_mu * C_rankmu)

    """Izvrši jednu generaciju CMA-ES algoritma. Vraća False ako je konvergirao."""    
    def step(self):
        B, D = self._decompose()
        x, y, _ = self._sample_population(B, D)

        # Evaluacija funkcije cilja
        fitness = np.array([self.f(xi) for xi in x])

        # Sortiranje po fitnesu (rastući redosled – minimizacija)
        order = np.argsort(fitness)
        x_sorted = x[order]
        y_sorted = y[order]
        f_sorted = fitness[order]

        # Ažuriranje najboljeg rešenja
        if f_sorted[0] < self.best_f:
            self.best_f = f_sorted[0]
            self.best_x = x_sorted[0].copy()

        # Ažuriranje parametara raspodele
        delta_m = self._update_m(x_sorted)
        self._update_sigma(delta_m, B, D)
        self._update_C(delta_m, y_sorted)

        # Snimanje istorije
        self.history["best_fitness"].append(self.best_f)
        self.history["mean_fitness"].append(np.mean(fitness))
        self.history["sigma"].append(self.sigma)
        self.history["best_x"].append(self.best_x.copy())

        self.generation += 1

        # Kriterijum zaustavljanja: sigma premala ili max iteracija
        return self.sigma > self.tol and self.generation < self.max_iter

    def run(self, verbose=True, print_every=50):
        """Pokretanje CMA-ES do konvergencije."""
        while self.step():
            if verbose and self.generation % print_every == 0:
                print(f"Gen {self.generation:4d} | "
                      f"best f = {self.best_f:.6e} | "
                      f"sigma = {self.sigma:.4e} | "
                      f"best x = {np.round(self.best_x, 4)}")

        if verbose:
            print(f"\n✓ Završeno u {self.generation} generacija")
            print(f"  Najbolji fitnes : {self.best_f:.6e}")
            print(f"  Najbolji x      : {np.round(self.best_x, 6)}")

        return self.best_x, self.best_f
