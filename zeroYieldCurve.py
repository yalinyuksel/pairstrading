#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 18:55:55 2025

@author: yalinyuksel
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Settings and helpers
# ---------------------------------------------------------------
SETTLE = datetime(2025, 5, 1)             # valuation date

def yearfrac(d1, d2, basis="ACT/365F"):
    """Simple ACT/365F year-fraction."""
    return (d2 - d1).days / 365.0

def price_from_ytm(cpn, freq, ytm, maturity, fv=100):
    """Dirty price when only YTM is given."""
    n = int(round(yearfrac(SETTLE, maturity) * freq))
    flow = cpn * fv / freq
    dfs  = [(1 + ytm / freq) ** -(i + 1) for i in range(n)]
    return sum(flow * d for d in dfs[:-1]) + (flow + fv) * dfs[-1]

# ---------------------------------------------------------------
# 2. Load your bond data
# ---------------------------------------------------------------
df = pd.read_csv("/Users/yalinyuksel/Desktop/QNB/Scripts/bond_portfolio.csv")

# Fill any missing dirty_price from YTM
df["dirty_price"] = df.apply(
    lambda r: r["dirty_price"]
              if pd.notna(r["dirty_price"])
              else price_from_ytm(r["coupon_rate"],
                                 int(r["frequency"]),
                                 r["ytm"],
                                 pd.to_datetime(r["maturity_date"])),
    axis=1
)

# Add year-fraction maturity pillar and sort short→long
df["T"] = pd.to_datetime(df["maturity_date"]).apply(lambda d: yearfrac(SETTLE, d))
df = df.sort_values("T").reset_index(drop=True)

# ---------------------------------------------------------------
# 3. Sequential bootstrap with Fix-2 interpolation
# ---------------------------------------------------------------
discount = {}          # {pillar T : DF}

for _, row in df.iterrows():
    cpn_rate = row["coupon_rate"]
    freq     = int(row["frequency"])
    fv       = 100
    n_period = int(round(row["T"] * freq))
    flow     = cpn_rate * fv / freq

    pv_known = 0.0
    for j in range(n_period - 1):          # all coupons except the final one
        t_j = round((j + 1) / freq, 8)     # round avoids 0.499999 vs 0.5 issues

        if t_j in discount:                               # already solved
            pv_known += flow * discount[t_j]
        else:
            keys = sorted(discount.keys())
            # ---------- Fix-2 block ----------
            if not keys or max(keys) < t_j:
                # No pillar to the right yet → use last known DF (piece-wise flat)
                if keys:                               # at least one DF exists
                    pv_known += flow * discount[keys[-1]]
                else:                                  # first bond, no DF yet
                    # leave pv_known += 0  (coupon treated as 0 for now)
                    pass
            else:
                # Standard linear interpolation between nearest left & right pillars
                left  = max(k for k in keys if k < t_j)
                right = min(k for k in keys if k > t_j)
                w     = (t_j - left) / (right - left)
                interp_df = discount[left] * (1 - w) + discount[right] * w
                pv_known  += flow * interp_df
            # ---------- end Fix-2 block ----------

    # Solve for DF at the bond’s maturity T
    P     = row["dirty_price"]
    DF_T  = (P - pv_known) / (flow + fv)
    discount[row["T"]] = DF_T

# ---------------------------------------------------------------
# 4. Convert to zero (spot) rates
# ---------------------------------------------------------------
zeros_annual = {T: DF ** (-1 / T) - 1 for T, DF in discount.items()}
zeros_cont   = {T: -np.log(DF) / T     for T, DF in discount.items()}

zero_curve = (
    pd.DataFrame({
        "T (yrs)"        : discount.keys(),
        "DiscountFactor" : discount.values(),
        "ZeroRate_ann"   : zeros_annual.values(),
        "ZeroRate_cont"  : zeros_cont.values()
    })
    .sort_values("T (yrs)")
    .reset_index(drop=True)
)

print("\nBootstrapped Zero Curve – first few pillars")
print(zero_curve.head(10).to_string(index=False))

# Optionally save to CSV for downstream use
# zero_curve.to_csv("zero_curve_out.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Assume `zero_curve` is already defined in your session, with columns:
#      - maturity in years: either "T" or "T (yrs)"
#      - zero rate: either "zero" or "ZeroRate_ann"
# ──────────────────────────────────────────────────────────────────────────────

# Detect column names
cols = zero_curve.columns.tolist()
if "T (yrs)" in cols:
    mat_col = "T (yrs)"
elif "T" in cols:
    mat_col = "T"
else:
    raise KeyError(f"Cannot find maturity column in {cols!r}")

if "ZeroRate_ann" in cols:
    rate_col = "ZeroRate_ann"
elif "zero" in cols:
    rate_col = "zero"
else:
    raise KeyError(f"Cannot find zero‐rate column in {cols!r}")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Define your standard tenors
# ──────────────────────────────────────────────────────────────────────────────
desired_tenors = [0.5] + list(range(1, 11))   # 0.5Y, 1Y, 2Y, …, 10Y

# ──────────────────────────────────────────────────────────────────────────────
# 3) Interpolate the zero rates
# ──────────────────────────────────────────────────────────────────────────────
zc = zero_curve.set_index(mat_col)
full_idx = zc.index.union(desired_tenors)

zc_interp = (
    zc.reindex(full_idx)
      .sort_index()
      .interpolate(method="linear")
      .loc[desired_tenors]
      .reset_index()
      .rename(columns={'index': mat_col})
)

# Optional: view the interpolated DataFrame
print(zc_interp[[mat_col, rate_col]].to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 4) Plot the interpolated curve
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(zc_interp[mat_col], zc_interp[rate_col], marker='o', linestyle='-')
plt.title("Interpolated Zero Curve (Annual Compounding)")
plt.xlabel("Maturity (years)")
plt.ylabel("Zero Rate")
plt.grid(True)
plt.tight_layout()
plt.show()