import datetime
import yfinance as yf
from src.pricing import black_scholes

TICKER = "GLD"  
ATM_BAND = 0.05  # show strikes within 5% of spot

def get_risk_free_rate() -> float:
    irx = yf.Ticker("^IRX")
    rate = irx.history(period="1d")["Close"].iloc[-1]
    return rate / 100

def time_to_expiry(expiry_str: str) -> float:
    expiry = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    today = datetime.date.today()
    return (expiry - today).days / 365

def main():
    ticker = yf.Ticker(TICKER)
    spot = ticker.history(period="1d")["Close"].iloc[-1]
    r = get_risk_free_rate()
    expiry = ticker.options[1]
    T = time_to_expiry(expiry)
    chain = ticker.option_chain(expiry)

    print(f"\n{TICKER}  spot={spot:.2f}  expiry={expiry}  T={T:.4f}yr  r={r:.4f}\n")

    for label, contracts in [("CALLS", chain.calls), ("PUTS", chain.puts)]:
        atm = contracts[
            (contracts["strike"] >= spot * (1 - ATM_BAND)) &
            (contracts["strike"] <= spot * (1 + ATM_BAND))
        ].copy()

        print(f"{'─'*70}")
        print(f"{label}")
        print(f"{'Strike':>8}  {'Mkt Mid':>8}  {'BS Price':>8}  {'Diff':>8}  {'Impl Vol':>8}")
        print(f"{'─'*70}")

        for _, row in atm.iterrows():
            K = row["strike"]
            mid = (row["bid"] + row["ask"]) / 2
            iv = row["impliedVolatility"]
            if iv <= 0 or mid <= 0:
                continue
            try:
                call_bs, put_bs = black_scholes(S=spot, t=0, T=T, K=K, r=r, q=0, sigma=iv)
                bs_price = call_bs if label == "CALLS" else put_bs
                diff = bs_price - mid
                print(f"{K:>8.2f}  {mid:>8.4f}  {bs_price:>8.4f}  {diff:>+8.4f}  {iv:>8.4f}")
            except ValueError:
                continue

        print()

if __name__ == "__main__":
    main()
