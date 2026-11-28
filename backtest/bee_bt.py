"""Faithful Python port of bee-ict-bot.pine for backtesting on 1-minute NAS100 data.

Emulates TradingView's broker: signals at bar close, orders work from the next bar,
intrabar path open->high->low->close (or open->low->high->close when open is nearer the low),
slippage on market/stop orders, commission per contract per side.
"""
import glob, math, sys, json
import numpy as np, pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "fd/pyfinancialdata/data/currencies/oanda/NAS100_USD"
NY = "America/New_York"


def load(years):
    fs = [f for y in years for f in sorted(glob.glob(f"{DATA}/{y}/*.csv"))]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert(NY)
    d = d.drop_duplicates("time").set_index("time").sort_index()
    return d[["open", "high", "low", "close"]]


def resample(m1, rule, offset=None):
    r = m1.resample(rule, label="left", closed="left", offset=offset)
    return r.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def pivots(h, l, n):
    """Pine ta.pivothigh/low(n, n): value appears on bar i for the pivot at i-n."""
    N = len(h); ph = np.full(N, np.nan); pl = np.full(N, np.nan)
    for i in range(2 * n, N):
        c = i - n
        if h[c] > h[c - n:c].max() and h[c] > h[c + 1:i + 1].max():
            ph[i] = h[c]
        if l[c] < l[c - n:c].min() and l[c] < l[c + 1:i + 1].min():
            pl[i] = l[c]
    return ph, pl


def htf_trend(b):
    h, l, c = b.high.values, b.low.values, b.close.values
    ph, pl = pivots(h, l, 3)
    lastH = lastL = np.nan; tr = 0; out = np.zeros(len(b), int)
    for i in range(len(b)):
        if not np.isnan(ph[i]): lastH = ph[i]
        if not np.isnan(pl[i]): lastL = pl[i]
        if not np.isnan(lastH) and c[i] > lastH: tr = 1
        if not np.isnan(lastL) and c[i] < lastL: tr = -1
        out[i] = tr
    return pd.Series(out, index=b.index)


def htf_fvg(b):
    h, l = b.high.values, b.low.values
    bt = bb = st = sb = np.nan; o = np.full((len(b), 4), np.nan)
    for i in range(2, len(b)):
        if l[i] > h[i - 2]: bt, bb = l[i], h[i - 2]
        if h[i] < l[i - 2]: st, sb = l[i - 2], h[i]
        o[i] = (bt, bb, st, sb)
    return pd.DataFrame(o, index=b.index, columns=["bt", "bb", "st", "sb"])


DEFAULTS = dict(
    biasMode="auto", usePD=False, useHtfFvg=False,
    swLen=3, mssWin=30, dispMult=1.0, needDisp=True, fvgAtr=0.1,
    entryMode="fvg", entryLvl="edge", fillWin=20,
    sess=(9 * 60 + 30, 11 * 60), hold=(9 * 60 + 30, 15 * 60 + 45),
    riskP=0.5, capital=50000.0, pv=2.0, tick=0.25, comm=0.62, slip_ticks=1,
    stopBuf=1.0, tpMode="liq", minRR=2.0, fixedR=3.0, usePart=True, tp1R=1.0,
    maxTrades=2, maxLosses=2, dayLossP=1.5, oneDone=False,
)


def run(m1, tf="5min", **kw):
    P = {**DEFAULTS, **kw}
    b = resample(m1, tf)
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    t = b.index; N = len(b)
    mins = (t.hour * 60 + t.minute).values
    inSess = (mins >= P["sess"][0]) & (mins < P["sess"][1])
    inHold = (mins >= P["hold"][0]) & (mins < P["hold"][1])
    dates = np.array(t.date)

    # macro bias: previous completed 4H bar (lookahead_on + [1])
    b4 = resample(m1, "4h")
    tr4 = htf_trend(b4).shift(1)
    bias = tr4.reindex(t.floor("4h")).fillna(0).values.astype(int)
    # daily (17:00 NY session) previous day mid
    sd1 = (m1.index - pd.Timedelta(hours=17)).normalize()
    dd = m1.groupby(sd1).agg({"high": "max", "low": "min"})
    pdm = ((dd.high + dd.low) / 2).shift(1)
    pdMid = pdm.reindex((t - pd.Timedelta(hours=17)).normalize()).values
    # 1H FVG zone
    hz = htf_fvg(resample(m1, "1h")).shift(1).reindex(t.floor("1h")).values

    tr_ = np.maximum(h - l, np.abs(h - np.r_[np.nan, c[:-1]]), np.abs(l - np.r_[np.nan, c[:-1]]))
    tr_[0] = h[0] - l[0]
    atr = pd.Series(tr_).ewm(alpha=1 / 14, adjust=False).mean().values  # Pine RMA
    ph, pl = pivots(h, l, P["swLen"])

    st = dict(lastPH=np.nan, lastPL=np.nan, ssl=None, bsl=None, sLow=np.nan, sHigh=np.nan,
              bTop=np.nan, bBot=np.nan, bBar=None, sTop=np.nan, sBot=np.nan, sBar=None,
              dB=None, dS=None, bullFor=None, bearFor=None, hB=None, hS=None)
    liqs = []  # [price, is_high]
    pend = None   # dict(side, e, s, tgt, q, tp1, bar)
    pos = None    # dict(side, q, e, s, tgt, tp1, part_done, be_from)
    exit_next_open = False
    net = 0.0; equity_curve = []; trades = []
    day = None; tdy = ldy = wdy = 0; netDay = 0.0
    prevPH = np.nan; prevPL = np.nan
    slip = P["slip_ticks"] * P["tick"]

    def pnl_close(side, entry, px, q):
        return (px - entry) * q * P["pv"] * side - q * P["comm"]

    for i in range(N):
        # ---------- intrabar order processing (orders placed on previous bars) ----------
        path = [o[i], h[i], l[i], c[i]] if (h[i] - o[i]) <= (o[i] - l[i]) else [o[i], l[i], h[i], c[i]]
        if exit_next_open and pos:
            px = o[i] - slip * pos["side"]
            pl_ = pnl_close(pos["side"], pos["e"], px, pos["q"]); net += pl_; pos["realized"] += pl_
            trades.append({**pos, "exit": px, "xt": t[i], "why": "session"}); pos = None
        exit_next_open = False
        if pend and not pos:
            e = pend["e"]; side = pend["side"]
            filled_at = None
            if pend["market"]:
                filled_at = (0, o[i] + slip * side)
            else:
                # walk path to find first touch of the limit
                prev = path[0]
                if (side == 1 and prev <= e) or (side == -1 and prev >= e):
                    filled_at = (0, prev)  # gap through limit -> fill at open
                else:
                    for k in range(1, 4):
                        lo_, hi_ = min(prev, path[k]), max(prev, path[k])
                        if lo_ <= e <= hi_:
                            filled_at = (k, e); break
                        prev = path[k]
            if filled_at:
                k0, px = filled_at
                pos = dict(side=side, q=pend["q"], q0=pend["q"], e=px, s=pend["s"], tgt=pend["tgt"], tp1=pend["tp1"],
                           part=P["usePart"] and pend["q"] >= 2, part_done=False, be_bar=None,
                           et=t[i], risk_usd=pend["risk_usd"], realized=0.0, net_before=pend["net_before"])
                pend = None; tdy += 1; fill_seg = k0
            else:
                fill_seg = None
        else:
            fill_seg = 0 if pos else None
        if pos:
            # exits along the path, starting from the segment where we are
            start = fill_seg if fill_seg is not None else 0
            prev = path[start] if start > 0 else (pos["e"] if pos["et"] == t[i] else path[0])
            side = pos["side"]
            stop = pos["e"] if (pos["part_done"] and pos["be_bar"] is not None and i > pos["be_bar"]) else pos["s"]
            done = False
            # gap at open beyond stop/target
            if pos["et"] != t[i]:
                if (side == 1 and o[i] <= stop) or (side == -1 and o[i] >= stop):
                    px = o[i] - slip * side
                    pl_ = pnl_close(side, pos["e"], px, pos["q"]); net += pl_; pos["realized"] += pl_
                    trades.append({**pos, "exit": px, "xt": t[i], "why": "stop"}); pos = None; done = True
            for k in range(start + 1, 4):
                if done or not pos: break
                a, z = prev, path[k]
                lo_, hi_ = min(a, z), max(a, z)
                events = []
                if lo_ <= stop <= hi_: events.append((abs(stop - a), "stop", stop))
                if pos["part"] and not pos["part_done"] and lo_ <= pos["tp1"] <= hi_: events.append((abs(pos["tp1"] - a), "tp1", pos["tp1"]))
                if lo_ <= pos["tgt"] <= hi_: events.append((abs(pos["tgt"] - a), "tp", pos["tgt"]))
                for _, kind, lvl in sorted(events):
                    if not pos: break
                    if kind == "stop":
                        px = lvl - slip * side
                        pl_ = pnl_close(side, pos["e"], px, pos["q"]); net += pl_; pos["realized"] += pl_
                        trades.append({**pos, "exit": px, "xt": t[i], "why": "stop" if stop == pos["s"] else "be"}); pos = None
                    elif kind == "tp1":
                        q1 = pos["q0"] // 2
                        pl_ = pnl_close(side, pos["e"], lvl, q1); net += pl_; pos["realized"] += pl_
                        pos["q"] -= q1; pos["part_done"] = True; pos["be_bar"] = i
                    else:
                        pl_ = pnl_close(side, pos["e"], lvl, pos["q"]); net += pl_; pos["realized"] += pl_
                        trades.append({**pos, "exit": lvl, "xt": t[i], "why": "tp"}); pos = None
                prev = z

        # entry commission (charged on fill) — applied once per fill
        # (folded here for simplicity: subtract when trade recorded)

        # ---------- bar-close logic (Pine script body) ----------
        if dates[i] != day:
            day = dates[i]; tdy = ldy = wdy = 0; netDay = net
        if i < 2:
            equity_curve.append(net); continue
        a_ = atr[i]
        # FVG
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= a_ * P["fvgAtr"]:
            st["bTop"], st["bBot"], st["bBar"] = l[i], h[i - 2], i
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= a_ * P["fvgAtr"]:
            st["sTop"], st["sBot"], st["sBar"] = l[i - 2], h[i], i
        if c[i] > o[i] and (c[i] - o[i]) >= a_ * P["dispMult"]: st["dB"] = i
        if c[i] < o[i] and (o[i] - c[i]) >= a_ * P["dispMult"]: st["dS"] = i
        # HTF zone touches
        bt_, bb_, st_, sb_ = hz[i]
        if not np.isnan(bt_) and l[i] <= bt_ and h[i] >= bb_: st["hB"] = i
        if not np.isnan(st_) and h[i] >= sb_ and l[i] <= st_: st["hS"] = i
        # pivots -> liquidity
        prevPH, prevPL = st["lastPH"], st["lastPL"]
        if not np.isnan(ph[i]): st["lastPH"] = ph[i]; liqs.append([ph[i], True])
        if not np.isnan(pl[i]): st["lastPL"] = pl[i]; liqs.append([pl[i], False])
        while len(liqs) > 40: liqs.pop(0)
        sweptS = sweptB = False
        for k in range(len(liqs) - 1, -1, -1):
            p_, hi = liqs[k]
            if hi and h[i] > p_:
                if c[i] < p_: sweptB = True
                liqs.pop(k)
            elif (not hi) and l[i] < p_:
                if c[i] > p_: sweptS = True
                liqs.pop(k)
        if sweptS: st["ssl"] = i; st["sLow"] = l[i]
        if sweptB: st["bsl"] = i; st["sHigh"] = h[i]
        if st["ssl"] is not None and i - st["ssl"] <= P["mssWin"]: st["sLow"] = min(st["sLow"], l[i])
        if st["bsl"] is not None and i - st["bsl"] <= P["mssWin"]: st["sHigh"] = max(st["sHigh"], h[i])
        # MSS (crossover uses previous lastPH value)
        xUp = (not np.isnan(st["lastPH"])) and c[i] > st["lastPH"] and (not np.isnan(prevPH)) and c[i - 1] <= prevPH
        xDn = (not np.isnan(st["lastPL"])) and c[i] < st["lastPL"] and (not np.isnan(prevPL)) and c[i - 1] >= prevPL
        bullM = st["ssl"] is not None and i - st["ssl"] <= P["mssWin"] and xUp and st["bullFor"] != st["ssl"] and \
            ((not P["needDisp"]) or (st["dB"] is not None and st["dB"] >= st["ssl"]))
        bearM = st["bsl"] is not None and i - st["bsl"] <= P["mssWin"] and xDn and st["bearFor"] != st["bsl"] and \
            ((not P["needDisp"]) or (st["dS"] is not None and st["dS"] >= st["bsl"]))
        if bullM: st["bullFor"] = st["ssl"]
        if bearM: st["bearFor"] = st["bsl"]

        flat = pos is None
        dayPnL = net - netDay
        guard = tdy < P["maxTrades"] and ldy < P["maxLosses"] and dayPnL > -P["capital"] * P["dayLossP"] / 100 and not (P["oneDone"] and wdy > 0)
        allowL = P["biasMode"] in ("both", "long") or (P["biasMode"] == "auto" and bias[i] == 1)
        allowS = P["biasMode"] in ("both", "short") or (P["biasMode"] == "auto" and bias[i] == -1)
        pdOkL = (not P["usePD"]) or (not np.isnan(pdMid[i]) and c[i] < pdMid[i])
        pdOkS = (not P["usePD"]) or (not np.isnan(pdMid[i]) and c[i] > pdMid[i])
        htfL = (not P["useHtfFvg"]) or (st["hB"] is not None and st["ssl"] is not None and st["hB"] >= st["ssl"] - P["mssWin"])
        htfS = (not P["useHtfFvg"]) or (st["hS"] is not None and st["bsl"] is not None and st["hS"] >= st["bsl"] - P["mssWin"])

        def signal(isL):
            sw = st["ssl"] if isL else st["bsl"]; ext = st["sLow"] if isL else st["sHigh"]
            fBar = st["bBar"] if isL else st["sBar"]
            edge = st["bTop"] if isL else st["sBot"]
            mid = (st["bTop"] + st["bBot"]) / 2 if isL else (st["sTop"] + st["sBot"]) / 2
            if P["entryMode"] == "close": e = c[i]
            elif fBar is not None and sw is not None and fBar >= sw: e = mid if P["entryLvl"] == "ce" else edge
            else: return None
            s = ext - P["stopBuf"] if isL else ext + P["stopBuf"]
            r = e - s if isL else s - e
            if not r > 0: return None
            if P["tpMode"] == "liq":
                cands = [p_ for p_, hi in liqs if (hi if isL else not hi) and ((p_ - e) if isL else (e - p_)) >= r * P["minRR"]]
                if not cands: return None
                tgt = min(cands) if isL else max(cands)
            else:
                tgt = e + r * P["fixedR"] if isL else e - r * P["fixedR"]
            q = math.floor(P["capital"] * P["riskP"] / 100 / (r * P["pv"]))
            if q < 1: return None
            tp1 = e + r * P["tp1R"] if isL else e - r * P["tp1R"]
            return dict(side=1 if isL else -1, e=e, s=s, tgt=tgt, q=q, tp1=tp1, bar=i, market=P["entryMode"] == "close",
                        risk_usd=r * q * P["pv"], net_before=net)

        if flat and inSess[i] and guard:
            sig = None
            if bullM and allowL and pdOkL and htfL: sig = signal(True)
            if sig is None and bearM and allowS and pdOkS and htfS: sig = signal(False)
            if sig: pend = sig  # replaces any older pending order (both sides)
        if flat and pend and pend["bar"] != i and (i - pend["bar"] > P["fillWin"] or not inSess[i]):
            pend = None
        if pos and not inHold[i]:
            exit_next_open = True
        equity_curve.append(net)
        # count closed trades for daily guard
        while trades and not trades[-1].get("counted"):
            tr = trades[-1]; tr["counted"] = True
            tr["pnl"] = tr["realized"] - tr["q0"] * P["comm"]  # entry-side commission
            net -= tr["q0"] * P["comm"]; equity_curve[-1] = net
            if tr["pnl"] > 0: wdy += 1
            elif tr["pnl"] < 0: ldy += 1
    return trades, pd.Series(equity_curve, index=t)


def stats(trades, eq, capital=50000.0):
    if not trades: return dict(n=0)
    pn = np.array([t["pnl"] for t in trades]); R = np.array([t["pnl"] / t["risk_usd"] for t in trades])
    gp, gl = pn[pn > 0].sum(), -pn[pn < 0].sum()
    curve = capital + eq; peak = curve.cummax(); dd = (peak - curve).max()
    ddp = ((peak - curve) / peak).max() * 100
    return dict(n=len(pn), win=round((pn > 0).mean() * 100, 1), net=round(pn.sum()), pf=round(gp / gl, 2) if gl else None,
                avgR=round(R.mean(), 3), maxDD=round(dd), maxDDpct=round(ddp, 1),
                longs=int(sum(t["side"] == 1 for t in trades)), shorts=int(sum(t["side"] == -1 for t in trades)),
                exits={k: int(sum(t["why"] == k for t in trades)) for k in ("tp", "stop", "be", "session")})
