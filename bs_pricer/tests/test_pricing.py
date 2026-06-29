from src.pricing import black_scholes, delta, gamma, vega, theta, rho

def test_hull_example():
    call, put = black_scholes(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(call - 4.76) < 0.01

def test_zerodiv():
    call, put = black_scholes(S=1, t=0, T=0, K=2, r=0.1, q=0, sigma=0.2)
    assert call == max(1-2,0)
    assert put == max(2-1,0)

def test_delta_hull_example():
    call_delta, put_delta = delta(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(call_delta - 0.7791) < 0.01
    assert abs(put_delta - (-0.2209)) < 0.01

def test_gamma_hull_example():
    g = gamma(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(g - 0.0500) < 0.01

def test_vega_hull_example():
    v = vega(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(v - 0.0881) < 0.01

def test_theta_hull_example():
    call_theta, put_theta = theta(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(call_theta - (-4.5591)) < 0.01
    assert abs(put_theta - (-0.7542)) < 0.01

def test_rho_hull_example():
    call_rho, put_rho = rho(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(call_rho - 0.1398) < 0.01
    assert abs(put_rho - (-0.0504)) < 0.01