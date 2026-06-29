import math
from scipy.stats import norm
def black_scholes (S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> tuple[float, float]:
   if S <= 0 or K <= 0 or sigma <= 0:
      raise ValueError("S, K, and sigma must be positive")
   if t < 0 or T < 0:
      raise ValueError("t and T must be non-negative")
   if T - t <= 0:
      return (max(S - K, 0), max(K - S, 0))
   
   d1 = 1/(sigma*math.sqrt(T-t))*(math.log(S/K) + (r - q + 0.5*sigma**2)*(T-t))
   d2 = d1 - sigma*math.sqrt(T-t)
   call = norm.cdf(d1)*S*math.exp(-q*(T-t)) - norm.cdf(d2)*K*math.exp(-r*(T-t))
   put = norm.cdf(-d2)*K*math.exp(-r*(T-t)) - norm.cdf(-d1)*S*math.exp(-q*(T-t))
   return (call, put)