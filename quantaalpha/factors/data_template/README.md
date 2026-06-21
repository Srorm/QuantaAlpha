# How to read files.
For example, if you want to read `daily_pv.h5`:
```Python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files.** The DataFrame has a MultiIndex `(datetime, instrument)`. Instruments are qlib symbols like `SH600000`, `SZ000001`, `BJ920693`. Universe = full A-share market (index benchmarks are excluded from the tradable grid). Dates 2016-01-04 onward, daily.

# Data files

| Filename       | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| "daily_pv.h5"  | Adjusted daily price/volume + liquidity + flow + chip + analyst + ownership + PIT fundamentals + universe/tradability masks. ALL fields a factor can use are the COLUMNS below. |

# How factors consume this data (IMPORTANT)
A factor is a single EXPRESSION. The calculator loads `daily_pv.h5`, replaces every `$col` token with that column, then evaluates. **You may ONLY reference the `$`-columns listed below.** Operators available: cross-sectional `RANK/ZSCORE/MEAN/STD/MAX/MIN/MEDIAN/SKEW/KURT/SCALE` (group by date), time-series `TS_*/DELTA/DELAY/EMA/SMA/WMA/DECAYLINEAR/TS_CORR/TS_COVARIANCE/REGBETA/REGRESI/COUNT/SUMIF/FILTER/WHERE/MACD/RSI` (group by instrument), and arithmetic `ADD/SUBTRACT/MULTIPLY/DIVIDE/LOG/EXP/ABS/INV/POW/SQRT/SIGN/AND/OR`. There is **no group-by-industry and no point-in-time merge operator** — anything needing those is precomputed as a column below.

## CRITICAL WARNINGS
1. **`LOG` is log1p, NOT natural log.** The engine implements `LOG(x) = ln(x + 1)` (function_lib.py). For large inputs (e.g. `$total_mv`) the `+1` is negligible and `LOG($total_mv)` is a fine size proxy, but for ratios near 1 it is materially wrong: do **NOT** write `LOG($total_share / DELAY($total_share,252))` for net-issuance — use the plain ratio `DIVIDE($total_share, DELAY($total_share,252))` (or `SUBTRACT(DIVIDE(...),1)`) instead. If you genuinely need natural log of a near-1 quantity, request the `LOG_TRUE`/`LN` operator — **a true-LOG patch (np.log without +1) is still TODO in function_lib.py.**
2. **PIT fundamentals are forward-filled and CONSTANT between quarterly announcements.** Short-window time-series operators (`DELTA`, `TS_STD`, `TS_PCTCHANGE` over <60 sessions) on `$roe`, `$np_yoy`, `$revenue_ttm`, `$eps_fcst`, `$holder_num`, `$pledge_ratio`, etc. are near-zero and meaningless. For these use cross-sectional ops (`RANK`, `ZSCORE`) or windows >= 252 sessions.
3. **NaN is structural for many columns** (northbound only for Connect stocks; margin only for margin-eligible; chip data may start late; analyst columns `$analyst_cnt/$eps_fcst/$eps_rev` only for covered names; `$pledge_ratio/$insider_net_60d` only when reported; valuation NaN for loss-makers; fundamentals NaN pre-IPO; `$days_since_ann` NaN before first report). NaNs are dropped from cross-sectional `RANK/ZSCORE`. Do not assume universal coverage.
4. **Mask columns (`$is_st`,`$in_universe`,`$suspended`,`$limit_state`,`$tradable_buy`,`$tradable_sell`,`$tradable`,`$in_csi300`,`$in_csi500`,`$in_csi1000`,`$has_buyback`) are integer 0/1/-1, never NaN.** Multiply a factor by a mask or use `FILTER`/`WHERE` to restrict the universe. NOTE: `$tradable*` are factor INPUTS for cross-sectional universe cleaning; they do **not** gate the backtest's actual fills (Qlib's exchange enforces its own limit_threshold).
5. **Adjustment basis.** `$open/$high/$low/$close/$vwap/$pre_close` are qfq-adjusted. `$open_raw/$close_raw` are UNADJUSTED — use these for limit-band logic and for anything compared against raw-priced snapshots. Chip cost `$cyq_cost_avg` is on the **RAW** basis (compare with `$close_raw`, NOT `$close`).

---
## Price / volume (adjusted unless noted)
$open: qfq-adjusted open price (CNY).
$high: qfq-adjusted high price (CNY).
$low: qfq-adjusted low price (CNY).
$close: qfq-adjusted close price (CNY).
$volume: qfq-consistent daily volume (shares).
$return: daily simple return = close/prev_close - 1.
$amount: daily turnover value (CNY); adjustment-invariant liquidity proxy.
$vwap: qfq-adjusted volume-weighted average price (CNY).
$pre_close: qfq-consistent previous close (CNY); overnight gap = SUBTRACT(DIVIDE($open,$pre_close),1).
$adj_factor: qfq adjustment factor (raw = adjusted / $adj_factor).
$open_raw: UNADJUSTED open (CNY); for limit-band / price-floor logic.
$close_raw: UNADJUSTED close (CNY); raw basis shared by cyq cost and analyst-target columns.

## Liquidity / turnover / size / valuation (daily, already PIT)
$turnover_rate: turnover rate (%) on total shares.
$turnover_rate_f: free-float turnover rate (%); cleaner liquidity signal.
$volume_ratio: 量比 = today volume / recent average.
$pe: static P/E (NaN/neg for loss-makers).
$pe_ttm: TTM P/E; earnings yield = DIVIDE(1,$pe_ttm).
$pb: price-to-book; book-to-price = DIVIDE(1,$pb).
$ps: static price-to-sales.
$ps_ttm: TTM price-to-sales; sales-to-price = DIVIDE(1,$ps_ttm).
$dv_ratio: dividend yield (%) (last annual).
$dv_ttm: TTM dividend yield (%).
$total_share: total shares (10k); net issuance = DIVIDE($total_share,DELAY($total_share,252)).
$float_share: float shares (10k).
$free_share: free-float shares (10k).
$total_mv: total market cap (10k CNY); size = LOG($total_mv) (log1p; fine here) or RANK($total_mv).
$circ_mv: circulating market cap (10k CNY).

## Money flow — order-size (大中小单), daily, NaN where unavailable
Ratios are normalized by the moneyflow per-day TOTAL (summed buy+sell across all order buckets), so they are unit-clean and bounded ~[-1,1] (NOT divided by $amount).
$mf_net_ratio: net main-force inflow / moneyflow total. Persistent inflow = TS_MEAN($mf_net_ratio,5).
$mf_net_amt: net main-force money inflow (CNY, level).
$mf_big_ratio: net large+xlarge order inflow / moneyflow total (institutional/big-order proxy).
$mf_xlg_ratio: net extra-large (super) order inflow / moneyflow total.
$mf_sml_ratio: net small-order inflow / moneyflow total (retail proxy). Inst-vs-retail divergence = SUBTRACT($mf_big_ratio,$mf_sml_ratio).

## Northbound / Stock-Connect (NaN for non-Connect names)
$north_hold_ratio: northbound holding as pct of FLOAT (vol/float_share; NOT free float).
$north_hold_chg: 5-session change in northbound holding ratio (computed on the trading-session axis).
$north_money_mkt: market-level net northbound flow (亿元), broadcast to all stocks (regime overlay).

## Margin trading 融资融券 (NaN for non-margin-eligible names)
$margin_fin_bal: financing balance 融资余额 (CNY).
$margin_short_bal: securities-lending balance 融券余额 (CNY).
$margin_fin_ratio: financing balance / circulating mcap (leverage intensity).
$margin_buy_ratio: daily financing-buy / turnover (financing-buy intensity).
$margin_net_ratio: (financing - short) / circ_mv (net leveraged-bull pressure).

## Chip distribution 筹码 (cyq, RAW price basis; may start late -> NaN early)
$cyq_winrate: fraction of holders in profit at today price (0..1). Contrarian: high -> overhang.
$cyq_cost_avg: weighted-average holding cost (RAW price units; compare with $close_raw).
$cyq_price_cost: SUBTRACT(DIVIDE($close_raw,$cyq_cost_avg),1) (profit ratio; both RAW basis — corrected from qfq rebase).
$cyq_concent: chip concentration (cost_85pct-cost_15pct)/cost_50pct; lower = more concentrated.

## Curated technical oscillators (qfq basis; additive vs OHLCV-derived technicals)
$mfi_q: Money Flow Index (0..100), volume-weighted typical-price flow (hard to reconstruct from OHLCV alone).
$cci_q: Commodity Channel Index (mean-deviation oscillator).
$wr_q: Williams %R (bounded overbought/oversold oscillator).
$dmi_adx: DMI ADX trend-strength (Wilder-smoothed; recursive, error-prone to derive).
$asi: Accumulation Swing Index (cumulative recursive swing).
$emv: Ease of Movement (volume-range coupled).
$trix: TRIX triple-EMA rate-of-change oscillator.
$obv: On-Balance Volume (cumulative signed volume; level series).
$atr: Average True Range (Wilder-smoothed volatility, qfq price units).

## PIT fundamentals — ratios (latest reported as of trade date, gated by f_ann_date, ffilled between quarters)
$roe: ROE (%).
$roe_ttm: TTM/weighted ROE (%), smoother.
$roa: ROA (%).
$roic: ROIC (%).
$np_margin: net profit margin (%).
$gp_margin: gross profit margin (%); Novy-Marx GP/A = DIVIDE(SUBTRACT($revenue_ttm,$oper_cost_ttm),$total_assets).
$debt_assets: debt-to-assets (%) (leverage).
$cur_ratio: current ratio.
$quick_ratio: quick ratio.
$asset_turn: total asset turnover (DuPont turnover leg).
$inv_turn: inventory turnover.
$ar_turn: receivables turnover.
$ocf_to_rev: operating cash flow / revenue (earnings quality).
$accrual: cash conversion (OCF / net profit); low/neg flags accruals risk (short side).

## PIT fundamentals — growth (YoY %, PIT)
$rev_yoy: revenue YoY growth.
$np_yoy: net-profit YoY growth.
$np_yoy_q: single-quarter net-profit YoY (sharper earnings momentum / acceleration).
$eps_yoy: basic-EPS YoY growth.
$op_yoy: operating-profit YoY growth.
$equity_yoy: book-value (equity) YoY growth.
$asset_yoy: total-asset YoY growth (CMA / asset-growth anomaly — contrarian, short large values).

## PIT fundamentals — per-share & TTM levels (CNY)
TTM aggregates are stamped with the MAX visibility-date over their four constituent quarters (no early leak from a late-announced quarter).
$dt_eps: deducted-non-recurring diluted EPS.
$bps: book value per share.
$ebit_ttm: TTM EBIT.
$ebitda_ttm: TTM EBITDA (元); EV/EBITDA = DIVIDE(SUBTRACT(ADD(MULTIPLY($total_mv,10000),$total_liab),$money_cap),$ebitda_ttm).
$revenue_ttm: TTM operating revenue.
$oper_cost_ttm: TTM operating cost (gross profit = SUBTRACT($revenue_ttm,$oper_cost_ttm)).
$op_profit_ttm: TTM operating profit.
$total_profit_ttm: TTM total profit (earnings-purity numerator).
$net_profit_ttm: TTM net profit attributable to parent (E for E/P, accruals, payout).
$ocf_ttm: TTM operating cash flow; OCF yield = DIVIDE($ocf_ttm,MULTIPLY($total_mv,10000)); FCF = SUBTRACT($ocf_ttm,$capex_ttm).
$capex_ttm: TTM capex; investment-to-assets = DIVIDE($capex_ttm,$total_assets).
$div_paid_ttm: TTM cash dividends+interest paid (payout = DIVIDE($div_paid_ttm,$net_profit_ttm)).
$equity_raise_ttm: TTM cash from equity issuance (external financing / CMA leg).

## PIT fundamentals — balance-sheet levels (CNY, latest reported, NOT TTM)
$total_assets: total assets.
$total_equity: total shareholder equity (book value; B/P from raw, EV, equity multiplier).
$total_liab: total liabilities.
$money_cap: cash & equivalents.

## Earnings surprise / preannouncement / report timing (PIT by ann_date)
$fcst_np_chg: earnings-preannouncement midpoint profit-change YoY (%).
$express_roe: earnings-flash ROE (%).
$express_np_yoy: earnings-flash net-profit YoY (%).
$days_since_ann: trading days since latest fundamental announcement (gates PEAD windows). NaN before first report.
$days_to_report: trading sessions until the next SCHEDULED report (publicly pre-announced dates only, >=0). NaN if none scheduled. Pre-earnings-window flag.

## Analyst consensus / estimate revision (report_rc; NaN for uncovered names)
$analyst_cnt: distinct sell-side orgs publishing in trailing ~90 days (coverage breadth).
$eps_fcst: consensus next-FY forecast EPS (CNY).
$eps_rev: consensus-EPS revision = current consensus / consensus ~90 sessions ago - 1 (revision momentum).

## Ownership / supply / insider (PIT / event; NaN or 0 where unreported)
$holder_num: number of shareholders (PIT, latest period).
$holder_num_chg: QoQ pct change in shareholder count; falling (negative) = concentration, often bullish.
$pledge_ratio: share-pledge ratio (high = tail risk).
$float_unlock_30d: upcoming lockup-expiry share pct in next ~30 sessions (supply overhang; lockup dates are public in advance, so this is a forward window, not look-ahead). Best-effort (bulk pull messy).
$insider_net_60d: net insider increase-minus-decrease share ratio over trailing ~60 sessions (best-effort).
$has_buyback: 1 if an active/announced buyback in trailing ~120 sessions else 0 (int8, never NaN).

## Industry / benchmark / factor-mimicking helpers
$sw_l1: integer-encoded SW2021 level-1 industry id (PIT). Grouping key (for future GROUP_* ops). NaN if unmapped.
$sw_l1_ret: cross-sectional mean $return of the stock's SW-L1 industry that day. Sector-relative momentum = SUBTRACT($return,$sw_l1_ret) (sector-neutral WITHOUT a group operator).
$bench_ret_300: CSI300 daily return (broadcast); beta = REGBETA($return,$bench_ret_300,60), idio-vol = TS_STD(REGRESI($return,$bench_ret_300,60),60).
$bench_ret_500: CSI500 daily return (broadcast); size-matched beta.
$bench_ret_1000: CSI1000 daily return (broadcast); small-cap beta.
$smb_ret: daily small-minus-big factor-mimicking return (broadcast); FF size leg.
$hml_ret: daily high-minus-low book-to-price factor-mimicking return (broadcast); FF value leg. Multi-factor idio-vol = TS_STD(REGRESI($return, <stack of $bench_ret_300,$smb_ret,$hml_ret>),60) — request a multi-regressor REGRESI if not yet available; with single-regressor REGRESI use each leg separately.

## Universe / tradability masks (int8, never NaN)
$in_universe: 1 if listed & not delisted & status L on date.
$list_age_days: trading days since IPO (seasoning filter, e.g. exclude <60).
$is_st: 1 if ST/*ST/delisting-flagged (exclude from universe).
$suspended: 1 if full-day suspended (untradeable).
$limit_state: 1=up-limit, -1=down-limit, 2=炸板(failed seal), 0=none.
$tradable_buy: 1 if buy-fillable (in_universe & not suspended & vol>0 & not locked-up at open).
$tradable_sell: 1 if sell-fillable (in_universe & not suspended & vol>0 & not locked-down at open).
$tradable: 1 if both buy- and sell-fillable (default symmetric mask).
$in_csi300: 1 if CSI300 constituent on date.
$in_csi500: 1 if CSI500 constituent on date.
$in_csi1000: 1 if CSI1000 constituent on date.

## Example factors
- Value: `RANK(INV($pe_ttm))`, `RANK(DIVIDE(1, $pb))`, `RANK(DIVIDE($ocf_ttm, MULTIPLY($total_mv, 10000)))`
- Quality: `RANK(DIVIDE(SUBTRACT($revenue_ttm,$oper_cost_ttm), $total_assets))`, `RANK($roe_ttm)`
- Growth: `RANK($np_yoy_q)`, `RANK(SUBTRACT($np_yoy_q, DELAY($np_yoy_q, 60)))`
- Net issuance (NOT LOG — log1p distorts near-1 ratios): `MULTIPLY(-1, DIVIDE($total_share, DELAY($total_share, 252)))`
- Reversal: `MULTIPLY(-1, TS_PCTCHANGE($close, 5))`, `MULTIPLY(-1, SUBTRACT(DIVIDE($open,$pre_close),1))`
- Alpha101-style: `MULTIPLY(-1, TS_CORR(RANK($vwap), RANK($volume), 5))`
- Liquidity (Amihud): `TS_MEAN(DIVIDE(ABS($return), $amount), 20)`
- Flow: `TS_MEAN($mf_big_ratio, 5)`, `SUBTRACT($mf_big_ratio, $mf_sml_ratio)`
- Northbound: `$north_hold_chg`
- Chip (RAW basis): `MULTIPLY(-1, $cyq_winrate)`, `$cyq_price_cost`
- Analyst revision: `RANK($eps_rev)`, `RANK($analyst_cnt)`, `RANK($eps_fcst)`
- Ownership: `MULTIPLY(-1, $holder_num_chg)`, `MULTIPLY(-1, $pledge_ratio)`
- Supply overhang: `MULTIPLY(-1, $float_unlock_30d)`
- Sector-neutral: `SUBTRACT($return, $sw_l1_ret)`
- Multi-factor beta: `REGBETA($return, $smb_ret, 60)`
- Tradable-masked: `MULTIPLY(RANK(INV($pe_ttm)), $tradable)`

## Note for the factor proposer
Use the exact `$`-tokens above. Column names are readable but collision-safe ONLY under a patched parser that substitutes `$`-tokens longest-first with word boundaries (`re.sub(r'(?<![A-Za-z0-9_])\$name(?![A-Za-z0-9_])', ...)`). **This parser patch is a hard prerequisite — the current naive `expr.replace(col[1:], ...)` loop will corrupt these names.** Three field-list sites (finalize_dataset fields, factor_calculator.get_stock_data fields, and the proposer system_prompt) must also be updated to expose the full column set, or the new columns are invisible to the backtest.

## $sw_l1 industry code map (SW2021 L1)
- 1: 农林牧渔
- 2: 基础化工
- 3: 钢铁
- 4: 有色金属
- 5: 电子
- 6: 家用电器
- 7: 食品饮料
- 8: 纺织服饰
- 9: 轻工制造
- 10: 医药生物
- 11: 公用事业
- 12: 交通运输
- 13: 房地产
- 14: 商贸零售
- 15: 社会服务
- 16: 综合
- 17: 建筑材料
- 18: 建筑装饰
- 19: 电力设备
- 20: 国防军工
- 21: 计算机
- 22: 传媒
- 23: 通信
- 24: 银行
- 25: 非银金融
- 26: 汽车
- 27: 机械设备
- 28: 煤炭
- 29: 石油石化
- 30: 环保
- 31: 美容护理
