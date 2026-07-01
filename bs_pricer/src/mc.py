import numpy as np

def mc_euro(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float, N: int = 100000, seed: int | None = None) -> tuple[float, float, float, float]:
   if seed is not None:
      np.random.seed(seed)
   Z = np.random.standard_normal(N)
   S_T = S*np.exp((r-q - 0.5*sigma**2)*(T-t) + sigma*np.sqrt(T-t)*Z)

   call_payoffs = np.maximum(S_T-K, 0)
   call_mc = np.exp(-r*(T-t)) * np.mean(call_payoffs)
   call_stderr = np.exp(-r*(T-t)) * np.std(call_payoffs) / np.sqrt(N)

   put_payoffs = np.maximum(K-S_T,0)
   put_mc = np.exp(-r*(T-t)) * np.mean(put_payoffs)
   put_stderr = np.exp(-r*(T-t)) * np.std(put_payoffs) / np.sqrt(N)

   return (call_mc, call_stderr, put_mc, put_stderr)