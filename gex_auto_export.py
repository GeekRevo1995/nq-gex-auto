#!/usr/bin/env python3
# =============================================================================
#  NQ / MNQ GEX AUTO-EXPORT  (headless, for GitHub Actions / cron)
# =============================================================================
#  Derived from the Colab "NQ/MNQ GEX Dashboard" script — same Black-Scholes
#  math, same QQQ->NQ scaling, same Gamma Flip/Call Wall/Put Wall logic.
#  This version has NO charts and NO interactive output. It just computes
#  0DTE GEX and writes a JSON file shaped for gex-ladder.html to fetch
#  directly (skips the manual Sierra Chart XML paste for auto-refresh runs).
#
#  Output shape (data/gex_nq_0dte.json):
#  {
#    "generatedAt": "...",
#    "nqSpot": 27950.25,
#    "qqqSpot": 682.10,
#    "scale": 40.98,
#    "levels": [
#      {"label": "HVL 0DTE", "price": 27960.0, "major": true, "color": "#f1c40f"},
#      {"label": "Call Resistance 0DTE", "price": 28010.0, "major": true, "color": "#3498db"},
#      {"label": "Put Support 0DTE", "price": 27890.0, "major": true, "color": "#e67e22"},
#      {"label": "GEX 1", "price": 27980.0, "major": false, "color": "#2ecc71"},
#      ...
#    ]
#  }
#
#  LIMITATION — read this:
#    This script only computes ONE ranked GEX metric per strike (like the
#    original Colab script). It does NOT reproduce a separate "BL" category
#    the way the Sierra Chart export you get from your friend does — that
#    export uses a different, unknown methodology. All ranked levels here
#    are labeled "GEX 1".."GEX N" by |GEX| magnitude, mixed sign. Treat this
#    as a second, independent opinion — not a drop-in replacement for the
#    original source until you've cross-checked it for a few sessions.
# =============================================================================

import json
import os
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

RISK_FREE_RATE = 0.045
VOLUME_WEIGHT = 0.40
CONTRACT_MULT = 100
TOP_N_LEVELS = int(os.environ.get("GEX_TOP_N", "10"))
OUT_PATH = os.environ.get("GEX_OUT_PATH", "data/gex_nq_0dte.json")


def get_spots():
    qqq = yf.Ticker("QQQ")
    qqq_spot = float(qqq.history(period="1d")["Close"].iloc[-1])
    nq = yf.Ticker("NQ=F")
    nq_hist = nq.history(period="1d")["Close"]
    nq_spot = float(nq_hist.iloc[-1]) if len(nq_hist) else qqq_spot * 41.0
    scale = nq_spot / qqq_spot
    return qqq_spot, nq_spot, scale


def gamma_bs(S, K, T, r, sigma):
    try:
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            return 0.0
        if T <= 0:
            T = 1 / 365
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception:
        return 0.0


def fetch_0dte_gex(ticker, spot, today):
    expirations = ticker.options
    todays_exp = [e for e in expirations if (pd.Timestamp(e) - today).days == 0]
    if not todays_exp:
        # fall back to the nearest upcoming expiration if there's no true 0DTE today
        todays_exp = expirations[:1] if expirations else []

    rows = []
    for exp in todays_exp:
        try:
            chain = ticker.option_chain(exp)
            T = max((pd.Timestamp(exp) - today).days, 0) / 365
            T = max(T, 1 / 365)
            for opt_type, df, sign in [("CALL", chain.calls, 1), ("PUT", chain.puts, -1)]:
                for _, row in df.iterrows():
                    strike = row["strike"]
                    oi = row.get("openInterest", 0) or 0
                    vol = row.get("volume", 0) or 0
                    iv = row.get("impliedVolatility", np.nan)
                    effective_oi = oi + vol * VOLUME_WEIGHT
                    if effective_oi <= 0:
                        continue
                    gam = gamma_bs(spot, strike, T, RISK_FREE_RATE, iv)
                    gex = sign * gam * effective_oi * CONTRACT_MULT * spot ** 2 * 0.01
                    rows.append([strike, opt_type, effective_oi, gex])
        except Exception:
            continue

    if not rows:
        return None
    return pd.DataFrame(rows, columns=["strike", "type", "effective_oi", "gex"])


def analyze(df):
    levels = df.groupby("strike")["gex"].sum().reset_index().sort_values("strike")
    levels["cum_gex"] = levels["gex"].cumsum()

    gamma_flip = None
    for i in range(1, len(levels)):
        prev, curr = levels.iloc[i - 1]["cum_gex"], levels.iloc[i]["cum_gex"]
        if prev < 0 <= curr:
            s0, s1 = levels.iloc[i - 1]["strike"], levels.iloc[i]["strike"]
            frac = -prev / (curr - prev) if (curr - prev) != 0 else 0
            gamma_flip = s0 + frac * (s1 - s0)
            break

    call_oi = df[df["type"] == "CALL"].groupby("strike")["effective_oi"].sum()
    put_oi = df[df["type"] == "PUT"].groupby("strike")["effective_oi"].sum()
    call_wall = call_oi.idxmax() if len(call_oi) else None
    put_wall = put_oi.idxmax() if len(put_oi) else None

    return levels, gamma_flip, call_wall, put_wall


def main():
    qqq_spot, nq_spot, scale = get_spots()
    ticker = yf.Ticker("QQQ")
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    df = fetch_0dte_gex(ticker, qqq_spot, today)
    out_levels = []

    if df is not None:
        levels, gamma_flip, call_wall, put_wall = analyze(df)

        if gamma_flip is not None:
            out_levels.append({"label": "HVL 0DTE", "price": round(gamma_flip * scale, 2), "major": True, "color": "#f1c40f"})
        if call_wall is not None:
            out_levels.append({"label": "Call Resistance 0DTE", "price": round(call_wall * scale, 2), "major": True, "color": "#3498db"})
        if put_wall is not None:
            out_levels.append({"label": "Put Support 0DTE", "price": round(put_wall * scale, 2), "major": True, "color": "#e67e22"})

        ranked = levels.copy()
        ranked["abs_gex"] = ranked["gex"].abs()
        ranked = ranked.sort_values("abs_gex", ascending=False).head(TOP_N_LEVELS)
        for i, (_, r) in enumerate(ranked.iterrows(), start=1):
            out_levels.append({
                "label": f"GEX {i}",
                "price": round(r["strike"] * scale, 2),
                "major": False,
                "color": "#2ecc71" if r["gex"] >= 0 else "#e74c3c",
            })

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "nqSpot": round(nq_spot, 2),
        "qqqSpot": round(qqq_spot, 2),
        "scale": round(scale, 4),
        "levels": out_levels,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(out_levels)} levels to {OUT_PATH}")


if __name__ == "__main__":
    main()
