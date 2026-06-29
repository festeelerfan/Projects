import math
from scipy.stats import norm

def _d1_d2(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> tuple[float, float]:
   if S <= 0 or K <= 0 or sigma <= 0:
      raise ValueError("S, K, and sigma must be positive")
   if t < 0 or T < 0:
      raise ValueError("t and T must be non-negative")
   tau = T-t
   d1 = 1/(sigma*math.sqrt(tau))*(math.log(S/K) + (r - q + 0.5*sigma**2)*(tau))
   d2 = d1 - sigma*math.sqrt(tau)
   return (d1, d2)

def black_scholes (S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> tuple[float, float]:
   if S <= 0 or K <= 0 or sigma <= 0:
      raise ValueError("S, K, and sigma must be positive")
   if t < 0 or T < 0:
      raise ValueError("t and T must be non-negative")
   tau = T-t
   if tau <= 0:
      return (max(S - K, 0), max(K - S, 0))

   d1, d2 = _d1_d2(S, t, T, K, r, q, sigma)
   call = S*math.exp(-q*tau)*norm.cdf(d1) - K*math.exp(-r*(T-t))*norm.cdf(d2)
   put = K*math.exp(-r*tau)*norm.cdf(-d2) - S*math.exp(-q*(T-t))*norm.cdf(-d1)
   return (call, put)

def delta(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> tuple[float, float]:
   d1 = _d1_d2(S, t, T, K, r, q, sigma)[0]
   call = math.exp(-q*(T-t))*norm.cdf(d1)
   put = math.exp(-q*(T-t))*(norm.cdf(d1)-1)
   return (call, put)

def gamma(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> float:
   tau = T - t
   d1 = _d1_d2(S, t, T, K, r, q, sigma)[0]
   gamma = math.exp(-q*tau)/(S*sigma*math.sqrt(tau))*norm.pdf(d1)
   return gamma

def vega(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> float:
   tau = T - t
   d1 = _d1_d2(S, t, T, K, r, q, sigma)[0]
   v = .01*S*math.exp(-q*tau)*math.sqrt(tau)*norm.pdf(d1)
   return v

def theta(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> tuple[float, float]:
   tau = T - t
   d1, d2 = _d1_d2(S, t, T, K, r, q, sigma)
   call = -((S*sigma*math.exp(-q*tau))/(2*math.sqrt(tau))*norm.pdf(d1)) - r*K*math.exp(-r*tau)*norm.cdf(d2) + q*S*math.exp(-q*tau)*norm.cdf(d1)
   put = -((S*sigma*math.exp(-q*tau))/(2*math.sqrt(tau))*norm.pdf(d1)) + r*K*math.exp(-r*tau)*norm.cdf(-d2) - q*S*math.exp(-q*tau)*norm.cdf(-d1)
   return (call, put)

def rho(S: float, t: float, T: float, K: float, r: float, q: float, sigma: float) -> tuple[float, float]:
   tau = T - t
   d2 = _d1_d2(S, t, T, K, r, q, sigma)[1]
   call = .01*K*tau*math.exp(-r*tau)*norm.cdf(d2)
   put = -.01*K*tau*math.exp(-r*tau)*norm.cdf(-d2)
   return (call, put)