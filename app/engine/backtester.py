from __future__ import annotations

from typing import Optional

from app.engine.market_feed import compute_macd_series, compute_sma_series
from app.engine.paper_broker import calc_fees
from app.models import (
    BacktestEquityPoint,
    BacktestResult,
    BacktestTradePoint,
    BacktestRequest,
    OrderSide,
)


def run_backtest(req: BacktestRequest, bars: list[dict], short_name: str = "") -> BacktestResult:
    """Run an A-share faithful daily backtest (T+1, 100-share lots, stamp tax/commission/transfer)."""
    if not bars:
        raise ValueError(f"未获取到 {req.stock_code} 的历史日线数据，无法运行回测")
    if req.strategy_type == "MA_CROSS" and req.fast_period >= req.slow_period:
        raise ValueError("双均线策略要求快线周期严格小于慢线周期")
    if req.strategy_type == "GRID" and req.grid_lower > 0 and req.grid_upper > 0 and req.grid_lower >= req.grid_upper:
        raise ValueError("网格策略上限价必须大于下限价")

    ordered = sorted(bars, key=lambda b: str(b["trade_date"]))
    closes = [float(b["close"]) for b in ordered]
    fast_ma = compute_sma_series(closes, req.fast_period, decimals=4)
    slow_ma = compute_sma_series(closes, req.slow_period, decimals=4)
    macd_series = compute_macd_series(closes, 12, 26, 9)

    start_idx = max(0, len(ordered) - req.days)
    eval_bars = ordered[start_idx:]
    first_close = float(eval_bars[0]["close"])
    if first_close <= 0:
        raise ValueError("历史行情收盘价异常，无法初始化回测")

    # Resolve grid bounds if strategy is GRID
    grid_lower = round(req.grid_lower, 2)
    grid_upper = round(req.grid_upper, 2)
    if req.strategy_type == "GRID":
        sample_closes = [float(b["close"]) for b in eval_bars[: min(20, len(eval_bars))]]
        mid = sum(sample_closes) / len(sample_closes)
        if grid_lower <= 0 and grid_upper <= 0:
            grid_lower = round(mid * 0.96, 2)
            grid_upper = round(mid * 1.04, 2)
        elif grid_lower <= 0:
            grid_lower = round(grid_upper * 0.92, 2)
        elif grid_upper <= 0:
            grid_upper = round(grid_lower * 1.08, 2)
        if grid_lower >= grid_upper:
            raise ValueError("网格上下限设置无效，请检查价格区间")

    cash = float(req.initial_cash)
    shares = 0
    today_bought = 0
    avg_cost = 0.0
    closed_trades = 0
    win_trades = 0
    peak_assets = cash
    max_drawdown_pct = 0.0

    equity_curve: list[BacktestEquityPoint] = []
    signals: list[BacktestTradePoint] = []

    for idx in range(start_idx, len(ordered)):
        bar = ordered[idx]
        tdate = str(bar["trade_date"])
        close = float(bar["close"])
        # T+1: previous day's bought shares become sellable today
        today_bought = 0
        sellable_qty = shares

        action: Optional[str] = None
        reason = ""

        # 1. Check stop-loss / take-profit guardrails when holding sellable position
        if sellable_qty > 0 and avg_cost > 0:
            pnl_pct = ((close - avg_cost) / avg_cost) * 100.0
            if req.stop_loss_pct > 0 and pnl_pct <= -req.stop_loss_pct:
                action = "SELL"
                reason = f"触发止损 ({pnl_pct:.2f}%)"
            elif req.take_profit_pct > 0 and pnl_pct >= req.take_profit_pct:
                action = "SELL"
                reason = f"触发止盈 (+{pnl_pct:.2f}%)"

        # 2. Check strategy signal if no risk exit fired
        if action is None:
            if req.strategy_type == "MA_CROSS" and idx >= 1:
                pf, ps = fast_ma[idx - 1], slow_ma[idx - 1]
                cf, cs = fast_ma[idx], slow_ma[idx]
                if pf is not None and ps is not None and cf is not None and cs is not None:
                    if pf <= ps and cf > cs:
                        action = "BUY"
                        reason = f"MA{req.fast_period}({cf:.2f})上穿MA{req.slow_period}({cs:.2f})"
                    elif pf >= ps and cf < cs:
                        action = "SELL"
                        reason = f"MA{req.fast_period}({cf:.2f})下穿MA{req.slow_period}({cs:.2f})"
            elif req.strategy_type == "MACD_CROSS" and idx >= 2:
                pdif, pdea, _ = macd_series[idx - 1]
                cdif, cdea, _ = macd_series[idx]
                if pdif <= pdea and cdif > cdea:
                    action = "BUY"
                    reason = f"MACD金叉 (DIF={cdif:.3f}, DEA={cdea:.3f})"
                elif pdif >= pdea and cdif < cdea:
                    action = "SELL"
                    reason = f"MACD死叉 (DIF={cdif:.3f}, DEA={cdea:.3f})"
            elif req.strategy_type == "GRID":
                if close <= grid_lower and shares == 0:
                    action = "BUY"
                    reason = f"触及网格下限 {grid_lower:.2f} 低吸"
                elif close >= grid_upper and sellable_qty > 0:
                    action = "SELL"
                    reason = f"触及网格上限 {grid_upper:.2f} 高抛"

        # 3. Execute A-share compliant trade
        if action == "BUY" and close > 0:
            cur_equity = cash + shares * close
            desired_pos_val = cur_equity * (req.position_pct / 100.0)
            target_spend = min(cash, max(0.0, desired_pos_val - shares * close))
            lots = int(target_spend // (close * 100.0))
            while lots > 0:
                amt = round(lots * 100 * close, 2)
                _, _, _, fee = calc_fees(OrderSide.BUY, amt)
                if amt + fee <= cash + 1e-6:
                    break
                lots -= 1

            buy_qty = lots * 100
            if buy_qty > 0:
                amt = round(buy_qty * close, 2)
                _, _, _, fee = calc_fees(OrderSide.BUY, amt)
                cash = round(cash - amt - fee, 2)
                total_cost = shares * avg_cost + amt + fee
                shares += buy_qty
                today_bought += buy_qty
                avg_cost = total_cost / shares
                eq_after = round(cash + shares * close, 2)
                signals.append(
                    BacktestTradePoint(
                        trade_date=tdate,
                        side="BUY",
                        price=round(close, 2),
                        quantity=buy_qty,
                        fee=fee,
                        reason=reason,
                        equity_after=eq_after,
                        pnl=None,
                    )
                )
        elif action == "SELL" and sellable_qty > 0:
            amt = round(sellable_qty * close, 2)
            _, _, _, fee = calc_fees(OrderSide.SELL, amt)
            realized_pnl = round(amt - fee - sellable_qty * avg_cost, 2)
            cash = round(cash + amt - fee, 2)
            shares -= sellable_qty
            if shares <= 0:
                shares = 0
                avg_cost = 0.0
            closed_trades += 1
            if realized_pnl > 0:
                win_trades += 1
            eq_after = round(cash + shares * close, 2)
            signals.append(
                BacktestTradePoint(
                    trade_date=tdate,
                    side="SELL",
                    price=round(close, 2),
                    quantity=sellable_qty,
                    fee=fee,
                    reason=reason,
                    equity_after=eq_after,
                    pnl=realized_pnl,
                )
            )

        strat_assets = round(cash + shares * close, 2)
        bench_assets = round(req.initial_cash * (close / first_close), 2)
        strat_ret_pct = round(((strat_assets - req.initial_cash) / req.initial_cash) * 100.0, 2)
        bench_ret_pct = round(((bench_assets - req.initial_cash) / req.initial_cash) * 100.0, 2)

        if strat_assets > peak_assets:
            peak_assets = strat_assets
        if peak_assets > 0:
            dd_pct = ((peak_assets - strat_assets) / peak_assets) * 100.0
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

        equity_curve.append(
            BacktestEquityPoint(
                trade_date=tdate,
                close=round(close, 2),
                strategy_assets=strat_assets,
                benchmark_assets=bench_assets,
                strategy_return_pct=strat_ret_pct,
                benchmark_return_pct=bench_ret_pct,
            )
        )

    final_pt = equity_curve[-1]
    win_rate = round((win_trades / closed_trades) * 100.0, 2) if closed_trades > 0 else 0.0
    excess_ret = round(final_pt.strategy_return_pct - final_pt.benchmark_return_pct, 2)

    if req.strategy_type == "MA_CROSS":
        params_summary = f"双均线 MA{req.fast_period}/MA{req.slow_period} · 仓位{req.position_pct:.0f}%"
    elif req.strategy_type == "MACD_CROSS":
        params_summary = f"MACD(12,26,9) · 仓位{req.position_pct:.0f}%"
    else:
        params_summary = f"网格 [{grid_lower:.2f}, {grid_upper:.2f}] · 仓位{req.position_pct:.0f}%"

    if req.stop_loss_pct > 0:
        params_summary += f" · 止损{req.stop_loss_pct:.1f}%"
    if req.take_profit_pct > 0:
        params_summary += f" · 止盈{req.take_profit_pct:.1f}%"

    return BacktestResult(
        stock_code=req.stock_code,
        short_name=short_name or req.stock_code,
        strategy_type=req.strategy_type,
        days=req.days,
        bar_count=len(eval_bars),
        start_date=str(eval_bars[0]["trade_date"]),
        end_date=str(eval_bars[-1]["trade_date"]),
        initial_cash=round(req.initial_cash, 2),
        final_assets=final_pt.strategy_assets,
        total_return_pct=final_pt.strategy_return_pct,
        benchmark_return_pct=final_pt.benchmark_return_pct,
        excess_return_pct=excess_ret,
        max_drawdown_pct=round(max_drawdown_pct, 2),
        win_rate_pct=win_rate,
        trade_count=len(signals),
        closed_trades=closed_trades,
        params_summary=params_summary,
        equity_curve=equity_curve,
        signals=signals,
    )
