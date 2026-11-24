"""Leveraged stETH loop backtest, single file.

Deposit stETH, borrow ETH, buy stETH, repeat. The question is what it costs to
defend the position during a peg wobble once you account for market depth:
most loop calculators sell collateral at the oracle price, here the rebalance
goes through a stETH/ETH x*y=k pool so slippage scales with size vs depth.

    python steth_loop.py                      -> output/equity.png, output/leverage_sweep.png
    python steth_loop.py --equity 5000 --seed 42 --depeg-prob 0.02
    pytest steth_loop.py                      -> runs the tests at the bottom

deps: numpy pandas matplotlib (pytest for the tests)

Caveats: synthetic market data (not calibrated), approximate Aave params,
a single pool depth number, daily step so no intraday liquidations.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SECONDS_PER_YEAR = 365 * 24 * 3600
OUT = Path("output")


# rate models

@dataclass(frozen=True)
class KinkedIRM:
    # two-slope model: flat-ish until u_opt, then steep
    base_rate: float = 0.0
    slope1: float = 0.027
    slope2: float = 0.80
    optimal_utilisation: float = 0.90
    reserve_factor: float = 0.15

    def __post_init__(self):
        if not 0 < self.optimal_utilisation < 1:
            raise ValueError("optimal_utilisation must be in (0, 1)")
        if not 0 <= self.reserve_factor < 1:
            raise ValueError("reserve_factor must be in [0, 1)")

    def borrow_rate(self, utilisation):
        u = np.clip(np.asarray(utilisation, dtype=float), 0.0, 1.0)
        u_opt = self.optimal_utilisation
        below = self.base_rate + self.slope1 * u / u_opt
        above = self.base_rate + self.slope1 + self.slope2 * (u - u_opt) / (1.0 - u_opt)
        return np.where(u <= u_opt, below, above)

    def supply_rate(self, utilisation):
        u = np.clip(np.asarray(utilisation, dtype=float), 0.0, 1.0)
        return self.borrow_rate(u) * u * (1.0 - self.reserve_factor)

    def rate_sensitivity(self, utilisation, du=0.01):
        # finite difference, good enough for a quick look
        return float(self.borrow_rate(utilisation + du) - self.borrow_rate(utilisation))


# rough Aave v3 mainnet params, last checked a while ago -- re-check on chain
AAVE_V3_WETH = KinkedIRM(slope1=0.027, slope2=0.80, optimal_utilisation=0.90, reserve_factor=0.15)
AAVE_V3_USDC = KinkedIRM(slope1=0.05, slope2=0.60, optimal_utilisation=0.92, reserve_factor=0.10)
AAVE_V3_WBTC = KinkedIRM(slope1=0.04, slope2=3.00, optimal_utilisation=0.45, reserve_factor=0.20)


@dataclass
class AdaptiveCurveIRM:
    """Morpho Blue style curve, simplified.

    rate(u) = rate_at_target * curve(u). The level drifts up while u > target and
    down while u < target. Stateful: call step() with elapsed seconds.
    """

    target_utilisation: float = 0.90
    rate_at_target: float = 0.04
    steepness: float = 4.0
    adjustment_speed: float = 50.0  # per year at full error (log space)
    min_rate_at_target: float = 0.001
    max_rate_at_target: float = 2.0

    def _curve(self, u):
        u = min(max(u, 0.0), 1.0)
        t = self.target_utilisation
        if u <= t:
            err = (u - t) / t
            return 1.0 + err * (1.0 - 1.0 / self.steepness)
        err = (u - t) / (1.0 - t)
        return 1.0 + err * (self.steepness - 1.0)

    def borrow_rate(self, utilisation):
        return self.rate_at_target * self._curve(utilisation)

    def step(self, utilisation, elapsed_seconds):
        t = self.target_utilisation
        err = (utilisation - t) / (t if utilisation <= t else 1.0 - t)
        growth = np.exp(self.adjustment_speed * err * elapsed_seconds / SECONDS_PER_YEAR)
        self.rate_at_target = float(
            np.clip(self.rate_at_target * growth, self.min_rate_at_target, self.max_rate_at_target)
        )
        return self.borrow_rate(utilisation)

# amm / slippage

@dataclass
class ConstantProductPool:
    """x*y=k pool (uni v2). x = risky asset, y = quote."""

    reserve_x: float
    reserve_y: float
    fee: float = 0.003

    @property
    def price(self):
        return self.reserve_y / self.reserve_x

    def quote_sell_x(self, dx):
        if dx <= 0:
            return 0.0
        dx_eff = dx * (1.0 - self.fee)
        return self.reserve_y * dx_eff / (self.reserve_x + dx_eff)

    def execution_price(self, dx):
        return self.quote_sell_x(dx) / dx if dx > 0 else self.price

    def slippage(self, dx):
        return 1.0 - self.execution_price(dx) / self.price

    def sell_x(self, dx):
        dy = self.quote_sell_x(dx)
        self.reserve_x += dx
        self.reserve_y -= dy
        return dy

    def depth_at(self, pct):
        # how much x can be sold before spot drops by pct
        # price after sale = k/x'^2, so pct drop <=> x'^2 = x^2/(1-pct)
        if not 0 < pct < 1:
            raise ValueError("pct must be in (0, 1)")
        x_after = self.reserve_x / np.sqrt(1.0 - pct)
        return float(x_after - self.reserve_x) / (1.0 - self.fee)


@dataclass
class ConcentratedPool:
    """Uni v3 pool with uniform liquidity L on [p_low, p_high].

    Deep inside the band, nothing outside it. Which is exactly the problem
    during a depeg: the liquidity disappears when you need it.
    """

    liquidity: float
    price: float
    p_low: float
    p_high: float
    fee: float = 0.0005

    def reserves(self):
        sp, spa, spb = np.sqrt(self.price), np.sqrt(self.p_low), np.sqrt(self.p_high)
        x = self.liquidity * (1.0 / sp - 1.0 / spb)
        y = self.liquidity * (sp - spa)
        return float(x), float(y)

    def quote_sell_x(self, dx):
        if dx <= 0:
            return 0.0
        dx_eff = dx * (1.0 - self.fee)
        sp = np.sqrt(self.price)
        spa = np.sqrt(self.p_low)
        max_dx = self.liquidity * (1.0 / spa - 1.0 / sp)  # absorbable before hitting p_low
        filled = min(dx_eff, max_dx)
        sp_new = 1.0 / (1.0 / sp + filled / self.liquidity)
        return float(self.liquidity * (sp - sp_new))

    def execution_price(self, dx):
        return self.quote_sell_x(dx) / dx if dx > 0 else self.price

    def slippage(self, dx):
        return 1.0 - self.execution_price(dx) / self.price


def impermanent_loss(price_ratio):
    # full-range v2 LP vs hodl, r = p_end/p_start. IL(2) ~ -5.7%, IL(5) ~ -25.5%
    r = np.asarray(price_ratio, dtype=float)
    return 2.0 * np.sqrt(r) / (1.0 + r) - 1.0


def max_safe_liquidation_size(pool, liquidation_bonus):
    """Biggest sale where slippage < liquidation bonus. Past that, liquidators
    lose money and stop showing up. Bisection, nothing fancy."""
    lo, hi = 0.0, pool.reserve_x * 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pool.slippage(mid) < liquidation_bonus:
            lo = mid
        else:
            hi = mid
    return lo

# backtest

@dataclass
class LoopConfig:
    leverage: float = 5.0  # collateral / equity
    staking_apr: float = 0.032
    lt: float = 0.93  # aave e-mode steth/eth
    liquidation_bonus: float = 0.01
    rebalance_hf: float = 1.05
    target_hf: float = 1.15
    gas_cost_eth: float = 0.01  # per rebalance, rough


def simulate_market(n_days=365, seed=7, util_mean=0.80, util_std=0.08,
                    peg_mean=1.0, peg_std=0.004, depeg_prob=0.004, depeg_size=-0.06):
    """Synthetic daily utilisation + stETH/ETH peg. Not calibrated, just plausible."""
    rng = np.random.default_rng(seed)
    noise = np.cumsum(rng.normal(0, 0.15, n_days)) / np.sqrt(np.arange(1, n_days + 1))
    util = np.clip(util_mean + util_std * noise, 0.3, 0.99)

    peg = np.empty(n_days)
    x = peg_mean
    for t in range(n_days):
        x += 0.2 * (peg_mean - x) + rng.normal(0, peg_std)
        if rng.random() < depeg_prob:
            x += depeg_size
        peg[t] = x
    return pd.DataFrame({"day": np.arange(n_days), "utilisation": util, "peg": peg})


def _set_pool_price(pool, price):
    # arbs reset the pool to the market peg overnight, keep k constant
    k = pool.reserve_x * pool.reserve_y
    pool.reserve_x = float(np.sqrt(k / price))
    pool.reserve_y = float(np.sqrt(k * price))


def _sell_collateral(qty_steth, peg, pool):
    if pool is None:
        return qty_steth * peg
    _set_pool_price(pool, peg)
    return pool.sell_x(qty_steth)


def backtest_loop(market, cfg, irm, equity_eth=100.0, pool=None):
    """Run the loop day by day, return a ledger DataFrame.

    pool: optional stETH/ETH ConstantProductPool (reserve_x = stETH, reserve_y = ETH).
    """
    coll = equity_eth * cfg.leverage  # stETH units
    debt = coll * market["peg"].iloc[0] - equity_eth  # ETH
    rows = []
    liquidated = False

    for r in market.itertuples():
        if liquidated:
            rows.append(dict(day=r.day, equity_eth=0.0, hf=0.0, borrow_apr=np.nan,
                             debt_eth=0.0, coll_eth=0.0, slippage_eth=0.0,
                             rebalanced=False, liquidated=True))
            continue

        borrow_apr = float(irm.borrow_rate(r.utilisation))
        coll *= 1.0 + cfg.staking_apr / 365.0
        debt *= 1.0 + borrow_apr / 365.0
        coll_eth = coll * r.peg
        hf = coll_eth * cfg.lt / debt if debt > 0 else np.inf
        slippage_eth = 0.0
        rebalanced = False

        if hf < 1.0:
            liquidated = True
            equity = max(coll_eth - debt * (1.0 + cfg.liquidation_bonus), 0.0)
        elif hf < cfg.rebalance_hf:
            # sell s ETH worth of stETH so that HF lands on target:
            # (coll_eth - s)*lt = (debt - s)*target
            s = (debt * cfg.target_hf - coll_eth * cfg.lt) / (cfg.target_hf - cfg.lt)
            s = min(max(s, 0.0), coll_eth)
            qty = s / r.peg
            proceeds = _sell_collateral(qty, r.peg, pool)
            slippage_eth = s - proceeds
            rebalanced = True
            coll -= qty
            debt -= proceeds
            debt += cfg.gas_cost_eth
            coll_eth = coll * r.peg
            hf = coll_eth * cfg.lt / debt if debt > 0 else np.inf
            equity = coll_eth - debt
        else:
            equity = coll_eth - debt

        rows.append(dict(day=r.day, equity_eth=equity, hf=hf, borrow_apr=borrow_apr,
                         debt_eth=debt, coll_eth=coll_eth, slippage_eth=slippage_eth,
                         rebalanced=rebalanced, liquidated=liquidated))
    return pd.DataFrame(rows)


def summarise(ledger, equity0):
    eq = ledger["equity_eth"]
    ret = eq.iloc[-1] / equity0 - 1.0
    daily = eq.pct_change().dropna()
    daily = daily[np.isfinite(daily)]
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "total_return": float(ret),
        "annualised_return": float((1 + ret) ** (365 / len(ledger)) - 1),
        "sharpe": sharpe,
        "max_drawdown": float(dd),
        "min_hf": float(ledger["hf"].replace(np.inf, np.nan).min()),
        "liquidated": bool(ledger["liquidated"].any()),
        "n_rebalances": int(ledger["rebalanced"].sum()),
        "total_slippage_eth": float(ledger["slippage_eth"].sum()),
    }

# charts + cli


def make_pool():
    # 20k stETH / 20k ETH, ~ what you'd find on a quiet day in the curve pool
    return ConstantProductPool(reserve_x=20_000.0, reserve_y=20_000.0, fee=0.0004)


def chart_equity(market, equity):
    cfg = LoopConfig(leverage=5.0)
    free = backtest_loop(market, cfg, AAVE_V3_WETH, equity)
    paid = backtest_loop(market, cfg, AAVE_V3_WETH, equity, pool=make_pool())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(free["day"], free["equity_eth"], label="rebalance at oracle price")
    ax.plot(paid["day"], paid["equity_eth"], label="rebalance through stETH/ETH pool")
    ax.set_xlabel("day")
    ax.set_ylabel("equity (ETH)")
    ax2 = ax.twinx()
    ax2.plot(free["day"], free["hf"], color="grey", alpha=0.4, lw=0.8)
    ax2.axhline(1.0, color="red", ls="--", lw=0.8)
    ax2.set_ylabel("health factor")
    ax2.set_ylim(0.9, 1.6)
    ax.set_title("5x stETH loop, one synthetic year")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=130)
    plt.close(fig)
    return summarise(free, equity), summarise(paid, equity)


def chart_leverage_sweep(market, equity):
    levs = [1.5, 2, 3, 4, 5, 6, 7, 8]
    ret_free, ret_paid = [], []
    for lev in levs:
        cfg = LoopConfig(leverage=lev)
        ret_free.append(summarise(backtest_loop(market, cfg, AAVE_V3_WETH, equity), equity)["annualised_return"])
        ret_paid.append(summarise(backtest_loop(market, cfg, AAVE_V3_WETH, equity, pool=make_pool()), equity)["annualised_return"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levs, [100 * r for r in ret_free], marker="o", label="no execution cost")
    ax.plot(levs, [100 * r for r in ret_paid], marker="o", label="through 20k/20k pool")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("leverage")
    ax.set_ylabel("annualised return (%)")
    ax.set_title("Return vs leverage: where market depth starts to bite")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "leverage_sweep.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--equity", type=float, default=1000.0, help="own capital in ETH")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--depeg-prob", type=float, default=0.01)
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    market = simulate_market(n_days=args.days, seed=args.seed, depeg_prob=args.depeg_prob)
    s_free, s_paid = chart_equity(market, args.equity)
    chart_leverage_sweep(market, args.equity)

    fmt = lambda d: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
    print("5x loop, oracle price :", fmt(s_free))
    print("5x loop, through pool :", fmt(s_paid))
    print(f"charts -> {OUT}/")


#  tests

def test_kink_is_continuous():
    u = AAVE_V3_WETH.optimal_utilisation
    assert abs(AAVE_V3_WETH.borrow_rate(u - 1e-9) - AAVE_V3_WETH.borrow_rate(u + 1e-9)) < 1e-6


def test_pool_slippage_grows_with_size():
    pool = ConstantProductPool(10_000, 10_000, fee=0.0)
    assert pool.slippage(10) < pool.slippage(100) < pool.slippage(1000)


def test_il_known_values():
    assert np.isclose(impermanent_loss(2.0), -0.0572, atol=1e-3)
    assert np.isclose(impermanent_loss(5.0), -0.2546, atol=1e-3)


def test_pool_costs_more_than_oracle():
    market = simulate_market(n_days=200, seed=3, depeg_prob=0.02)
    cfg = LoopConfig(leverage=5.0)
    free = backtest_loop(market, cfg, AAVE_V3_WETH, 1000.0)
    paid = backtest_loop(market, cfg, AAVE_V3_WETH, 1000.0, pool=ConstantProductPool(20_000, 20_000, fee=0.0004))
    assert paid["slippage_eth"].sum() >= free["slippage_eth"].sum()
    assert paid["equity_eth"].iloc[-1] <= free["equity_eth"].iloc[-1]


def test_no_leverage_never_liquidated():
    market = simulate_market(n_days=365, seed=1, depeg_prob=0.05)
    ledger = backtest_loop(market, LoopConfig(leverage=1.0), AAVE_V3_WETH, 100.0)
    assert not ledger["liquidated"].any()


if __name__ == "__main__":
    main()
