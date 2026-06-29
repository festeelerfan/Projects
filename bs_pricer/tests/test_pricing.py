from src.pricing import black_scholes

def test_hull_example():
    call, put = black_scholes(S=42, t=0, T=0.5, K=40, r=0.1, q=0, sigma=0.2)
    assert abs(call - 4.76) < 0.01

def test_zerodiv():
    call, put = black_scholes(S=1, t=0, T=0, K=2, r=0.1, q=0, sigma=0.2)
    assert call == max(1-2,0)
    assert put == max(2-1,0)