"""
pachart_rt.py  —  Standalone 5-Second Live Chart  (v3)
=======================================================
New in v3:
  - Right-side dashboard with two resizable panels:
      • Transaction History  — live table of all trades
      • Event / Error Log    — tail of the bot's JSON log, colour-coded
  - Horizontal drag-handle between the two dashboard panels
  - Vertical   drag-handle between chart and dashboard (sidebar width)
  - Separate /api/dashboard_data endpoint (polls every 5 s independently)

Run:
    python pachart_rt.py
Then open:
    http://localhost:5055
"""

import os
import glob
import sqlite3
import gzip
import json
import pathlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, Response, render_template_string
from dotenv import load_dotenv

load_dotenv()

BASE_URL         = os.getenv("BASE_URL",            "https://api.india.delta.exchange")
DB_PATH          = os.getenv("DB_PATH",             "ha_struct.db")
LOG_DIR          = os.getenv("LOG_DIR",             "logs_struct")
PORT             = int(os.getenv("FAST_CHART_PORT", 5055))
HOST             = os.getenv("CHART_HOST",          "127.0.0.1")
VWAP_RESET_HOURS = int(os.getenv("VWAP_RESET_HOURS", 4))
LOG_TAIL_LINES   = int(os.getenv("LOG_TAIL_LINES",   200))

STATIC_DIR = pathlib.Path("static")
LW_CDN     = "https://cdnjs.cloudflare.com/ajax/libs/lightweight-charts/4.1.3/lightweight-charts.standalone.production.js"
LW_LOCAL   = STATIC_DIR / "lw-charts.js"

def ensure_local_lib():
    STATIC_DIR.mkdir(exist_ok=True)
    if LW_LOCAL.exists(): return
    try:
        print("[LIB] Downloading lightweight-charts...")
        LW_LOCAL.write_bytes(requests.get(LW_CDN, timeout=30).content)
    except Exception as e:
        print(f"Library download failed: {e}")
        raise SystemExit(1)

ensure_local_lib()
app = Flask(__name__, static_folder="static")

# ==========================================
# 1. LIVE DATA FETCHING & INDICATORS
# ==========================================

def get_live_candles(minutes=400):
    now    = datetime.now(timezone.utc)
    params = {
        "symbol": "BTCUSD", "resolution": "1m",
        "start":  int((now - timedelta(minutes=minutes)).timestamp()),
        "end":    int(now.timestamp()),
    }
    r    = requests.get(f"{BASE_URL}/v2/history/candles", params=params, timeout=5)
    data = r.json().get("result")
    if not data: return pd.DataFrame()
    df = (pd.DataFrame(data)
          .rename(columns={"t": "time", "o": "open", "h": "high",
                           "l": "low",  "c": "close", "v": "volume"})
          .astype(float))
    df["time"] = df["time"].astype(int)
    return df.sort_values("time").reset_index(drop=True)

def compute_heikin_ashi(df):
    ha             = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    hc             = ha["ha_close"].values
    ho             = np.empty(len(df))
    ho[0]          = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ho[i] = (ho[i - 1] + hc[i - 1]) * 0.5
    ha["ha_open"]  = ho
    ha["ha_high"]  = ha[["high",  "ha_open", "ha_close"]].max(axis=1)
    ha["ha_low"]   = ha[["low",   "ha_open", "ha_close"]].min(axis=1)
    ha["ha_green"] = ha["ha_close"] > ha["ha_open"]
    return ha

def compute_vwap_bands(df):
    now       = datetime.now(timezone.utc)
    hb        = (now.hour // VWAP_RESET_HOURS) * VWAP_RESET_HOURS
    sess_ts   = int(now.replace(hour=hb, minute=0, second=0, microsecond=0).timestamp())
    band_cols = ("vwap", "vwap_upper1", "vwap_lower1", "vwap_upper2", "vwap_lower2")

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]        = df["typical_price"] * df["volume"]
    df["in_session"]    = df["time"] >= sess_ts
    s                   = df[df["in_session"]].copy()

    if len(s) < 2:
        for c in band_cols: df[c] = np.nan
        return df

    s["cum_tp_vol"]  = s["tp_vol"].cumsum()
    s["cum_vol"]     = s["volume"].cumsum()
    s["vwap"]        = s["cum_tp_vol"] / s["cum_vol"]
    s["cum_sq_dev"]  = ((s["typical_price"] - s["vwap"]) ** 2 * s["volume"]).cumsum()
    s["std_dev"]     = np.sqrt(s["cum_sq_dev"] / s["cum_vol"])
    s["vwap_upper1"] = s["vwap"] + 0.8 * s["std_dev"]
    s["vwap_lower1"] = s["vwap"] - 0.8 * s["std_dev"]
    s["vwap_upper2"] = s["vwap"] + 1.6 * s["std_dev"]
    s["vwap_lower2"] = s["vwap"] - 1.6 * s["std_dev"]

    for c in band_cols:
        df[c] = np.nan
        df.loc[df["in_session"], c] = s[c].values
    return df

# ==========================================
# 2. DB OVERLAY  (chart markers + SL trail)
# ==========================================

def get_db_overlay_data():
    trades, active_sl, pivot_map, sl_history = [], None, {}, []
    if not os.path.exists(DB_PATH):
        return trades, active_sl, pivot_map, sl_history
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT candle_time, side, entry, stop FROM trades ORDER BY candle_time ASC"
        ).fetchall():
            trades.append(dict(r))
        for r in conn.execute(
            "SELECT time, is_pivot FROM candles WHERE is_pivot IS NOT NULL"
        ).fetchall():
            pivot_map[int(r["time"])] = r["is_pivot"]
        row = conn.execute(
            "SELECT active_sl FROM candles WHERE active_sl IS NOT NULL ORDER BY time DESC LIMIT 1"
        ).fetchone()
        if row: active_sl = float(row["active_sl"])
        for r in conn.execute(
            "SELECT time, active_sl FROM candles WHERE active_sl IS NOT NULL ORDER BY time ASC"
        ).fetchall():
            sl_history.append({"time": int(r["time"]), "value": float(r["active_sl"])})
        conn.close()
    except Exception as e:
        print(f"DB Overlay Error: {e}")
    return trades, active_sl, pivot_map, sl_history

# ==========================================
# 3. DASHBOARD DATA
# ==========================================

def get_trade_history():
    if not os.path.exists(DB_PATH): return []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, cycle_id, timestamp, side, entry, stop, size, status "
            "FROM trades ORDER BY id DESC"
        ).fetchall()
        conn.close()
        out = []
        for r in rows:
            t = dict(r)
            try:
                dt = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
                t["ts_fmt"] = dt.strftime("%m-%d %H:%M:%S")
            except Exception:
                t["ts_fmt"] = (t.get("timestamp") or "—")[:19]
            out.append(t)
        return out
    except Exception as e:
        print(f"Trade History Error: {e}")
        return []

def get_log_entries():
    """Read bot JSON log files, return newest-first list capped at LOG_TAIL_LINES."""
    pattern = os.path.join(LOG_DIR, "struct_*.json")
    files   = sorted(glob.glob(pattern), reverse=True)
    if not files: return []

    collected = []
    for fpath in files[:2]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        collected.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    collected = collected[-LOG_TAIL_LINES:]
    collected.reverse()   # newest first

    entries = []
    for obj in collected:
        ts_raw = obj.get("timestamp", "")
        try:
            dt     = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts_fmt = dt.strftime("%H:%M:%S")
        except Exception:
            ts_fmt = ts_raw[:8]
        entries.append({
            "ts":      ts_fmt,
            "level":   obj.get("level",   "INFO"),
            "section": obj.get("section", ""),
            "message": obj.get("message", ""),
            "extra":   {k: v for k, v in obj.items()
                        if k not in ("timestamp", "level", "section", "message", "cycle_id")}
        })
    return entries

# ==========================================
# 4. FLASK ROUTES
# ==========================================

def gzip_json(data):
    c = gzip.compress(json.dumps(data).encode("utf-8"), compresslevel=1)
    return Response(c, content_type="application/json",
                    headers={"Content-Encoding": "gzip"})

@app.route("/api/live_chart_data")
def api_live_chart_data():
    try:
        df = get_live_candles(minutes=400)
        if df.empty: return jsonify({"error": "No data"}), 500
        df = compute_heikin_ashi(df)
        df = compute_vwap_bands(df)
        bot_trades, current_bot_sl, pivot_map, sl_history = get_db_overlay_data()

        payload = {
            "ha": [], "vwap": [],
            "vwap_upper1": [], "vwap_lower1": [],
            "vwap_upper2": [], "vwap_lower2": [],
            "trail_sl":    sl_history,
            "markers":     [],
            "latest_price": float(df.iloc[-1]["close"]),
            "latest_sl":    current_bot_sl,
        }
        for _, row in df.iterrows():
            t = int(row["time"])
            payload["ha"].append({
                "time": t, "open": row["ha_open"], "high": row["ha_high"],
                "low":  row["ha_low"], "close": row["ha_close"],
            })
            if not pd.isna(row["vwap"]):
                payload["vwap"].append({"time": t, "value": row["vwap"]})
            for col in ("vwap_upper1", "vwap_lower1", "vwap_upper2", "vwap_lower2"):
                if not pd.isna(row[col]):
                    payload[col].append({"time": t, "value": row[col]})
            if t in pivot_map:
                lbl  = pivot_map[t]
                lo   = lbl.endswith("L")
                payload["markers"].append({
                    "time": t, "position": "belowBar" if lo else "aboveBar",
                    "color": "#4da6ff" if lo else "#fb923c",
                    "shape": "circle", "text": lbl, "size": 1,
                })
        for tr in bot_trades:
            buy = tr["side"] == "buy"
            payload["markers"].append({
                "time":     tr["candle_time"],
                "position": "belowBar" if buy else "aboveBar",
                "color":    "#00d4aa" if buy else "#ff4d6d",
                "shape":    "arrowUp" if buy else "arrowDown",
                "text":     f"{'LONG' if buy else 'SHORT'} @ {tr['entry']}",
                "size":     2,
            })
        payload["markers"] = sorted(payload["markers"], key=lambda x: x["time"])
        return gzip_json(payload)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/dashboard_data")
def api_dashboard_data():
    try:
        return gzip_json({
            "trades": get_trade_history(),
            "log":    get_log_entries(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 5. HTML FRONTEND
# ==========================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTCUSD · HA Live Dashboard</title>
<script src="/static/lw-charts.js"></script>
<style>
/* ── Reset & Root ─────────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e14;--bg2:#0d1320;--bg3:#0f1826;
  --border:#1e2d40;--text:#c9d1d9;--muted:#4a5568;
  --green:#00d4aa;--red:#ff4d6d;--blue:#4da6ff;
  --yellow:rgba(245,200,66,.85);--warn:#f59e0b;
  --topbar:50px;--ph:34px;
}

/* ── Global ───────────────────────────────────────────────────────── */
body{height:100vh;overflow:hidden;background:var(--bg);color:var(--text);
     font-family:'Consolas','Courier New',monospace;display:flex;flex-direction:column}

/* ── Topbar ───────────────────────────────────────────────────────── */
#topbar{flex:0 0 var(--topbar);background:var(--bg2);display:flex;align-items:center;
        padding:0 16px;gap:22px;border-bottom:1px solid var(--border);user-select:none}
.tag{font-family:sans-serif;font-weight:800;font-size:16px;color:#fff}
.badge{font-size:10px;background:rgba(0,212,170,.12);border:1px solid var(--green);
       padding:3px 8px;border-radius:4px;color:var(--green);letter-spacing:1px;
       animation:pulse 2s infinite}
.stat{display:flex;flex-direction:column;gap:1px}
.stat-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.stat-val{font-size:13px;font-weight:700}
#legend{display:flex;gap:14px;align-items:center;margin-left:auto;font-size:10px;color:var(--muted)}
.leg{display:flex;align-items:center;gap:5px}
.leg-line{width:18px;height:2px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

/* ── Main layout ──────────────────────────────────────────────────── */
#main{flex:1;display:flex;overflow:hidden;min-height:0}

/* ── Chart ────────────────────────────────────────────────────────── */
#chart-wrap{flex:1;min-width:200px;position:relative;overflow:hidden}
#chart{width:100%;height:100%}

/* ── Vertical drag handle (chart ↔ sidebar) ───────────────────────── */
#vresize{
  flex:0 0 5px;background:var(--border);cursor:col-resize;user-select:none;
  transition:background .15s;display:flex;align-items:center;justify-content:center
}
#vresize:hover,#vresize.drag{background:var(--blue)}
#vresize::after{content:'⋮';color:var(--muted);font-size:14px;pointer-events:none}

/* ── Sidebar ──────────────────────────────────────────────────────── */
#sidebar{
  width:360px;min-width:220px;max-width:680px;
  display:flex;flex-direction:column;
  background:var(--bg2);border-left:1px solid var(--border);overflow:hidden
}

/* ── Panel shared ─────────────────────────────────────────────────── */
.panel{display:flex;flex-direction:column;overflow:hidden;min-height:80px}
.ph{
  flex:0 0 var(--ph);display:flex;align-items:center;justify-content:space-between;
  padding:0 12px;background:var(--bg3);border-bottom:1px solid var(--border);
  font-size:11px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
  color:var(--muted);user-select:none
}
.ph span{color:var(--text)}
.pbadge{
  font-size:10px;padding:2px 8px;border-radius:10px;font-weight:700;letter-spacing:.3px
}
.pbody{
  flex:1;overflow-y:auto;overflow-x:hidden;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent
}
.pbody::-webkit-scrollbar{width:5px}
.pbody::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── Horizontal drag handle (panel ↔ panel) ───────────────────────── */
#hresize{
  flex:0 0 5px;background:var(--border);cursor:row-resize;user-select:none;
  transition:background .15s;display:flex;align-items:center;justify-content:center
}
#hresize:hover,#hresize.drag{background:var(--blue)}
#hresize::after{content:'···';color:var(--muted);font-size:12px;letter-spacing:2px;pointer-events:none}

/* ── Transaction table ────────────────────────────────────────────── */
#tx-table{width:100%;border-collapse:collapse;font-size:11px}
#tx-table thead th{
  position:sticky;top:0;background:var(--bg3);padding:6px 8px;text-align:left;
  font-size:9px;text-transform:uppercase;letter-spacing:.5px;
  color:var(--muted);border-bottom:1px solid var(--border);font-weight:700
}
#tx-table tbody tr{border-bottom:1px solid rgba(30,45,64,.6);transition:background .1s}
#tx-table tbody tr:hover{background:rgba(77,166,255,.06)}
#tx-table td{padding:6px 8px;vertical-align:middle}
.sl{color:var(--green);font-weight:700}
.ss{color:var(--red);font-weight:700}
.sp{color:var(--blue)}
.sf{color:var(--green)}
.sc{color:var(--muted)}
.empty{padding:24px;text-align:center;color:var(--muted);font-size:12px;line-height:1.8}

/* ── Log list ─────────────────────────────────────────────────────── */
#log-list{list-style:none;font-size:11px}
#log-list li{
  display:flex;gap:7px;align-items:flex-start;
  padding:5px 10px;border-bottom:1px solid rgba(30,45,64,.5);line-height:1.45
}
#log-list li:hover{background:rgba(255,255,255,.02)}
.lts{flex:0 0 56px;color:var(--muted);font-size:10px;padding-top:1px}
.lsec{flex:0 0 64px;font-size:9px;text-transform:uppercase;letter-spacing:.4px;
       color:var(--blue);padding-top:2px}
.lmsg{flex:1;word-break:break-word}
.lW{color:var(--warn)}
.lE{color:var(--red)}
.lW .lmsg,.lE .lmsg{font-weight:600}
.lext{font-size:10px;color:var(--muted);margin-top:2px}

/* ── Follow button ────────────────────────────────────────────────── */
.fbtn{
  font-size:9px;padding:2px 7px;border-radius:3px;border:none;cursor:pointer;
  font-family:inherit;letter-spacing:.3px;
  background:rgba(77,166,255,.15);color:var(--blue);transition:background .15s
}
.fbtn.on{background:var(--blue);color:#000}
</style>
</head>
<body>

<!-- ═══ TOPBAR ═══════════════════════════════════════════════════════ -->
<div id="topbar">
  <span class="tag">BTCUSD</span>
  <span class="badge">LIVE 5s</span>

  <div class="stat">
    <span class="stat-label">Price</span>
    <span class="stat-val" id="tb-price" style="color:#fff">—</span>
  </div>
  <div class="stat">
    <span class="stat-label">Bot SL</span>
    <span class="stat-val" id="tb-sl" style="color:var(--red)">—</span>
  </div>
  <div class="stat">
    <span class="stat-label">Status</span>
    <span class="stat-val" id="tb-status" style="color:var(--muted);font-size:11px">Connecting…</span>
  </div>

  <div id="legend">
    <label class="leg" style="cursor:pointer;user-select:none">
      <input type="checkbox" id="vwap-toggle" checked style="cursor:pointer;width:14px;height:14px">
      <div class="leg-line" style="background:var(--yellow)"></div>VWAP
    </label>
    <label class="leg" style="cursor:pointer;user-select:none">
      <input type="checkbox" id="vwap-band1-toggle" style="cursor:pointer;width:14px;height:14px">
      <div class="leg-line" style="background:rgba(245,200,66,.4);border-top:1px dashed rgba(245,200,66,.4)"></div>±0.8σ
    </label>
    <label class="leg" style="cursor:pointer;user-select:none">
      <input type="checkbox" id="vwap-band2-toggle" style="cursor:pointer;width:14px;height:14px">
      <div class="leg-line" style="background:rgba(245,200,66,.18);border-top:1px dashed rgba(245,200,66,.18)"></div>±1.6σ
    </label>
    <div class="leg"><div class="leg-line" style="background:var(--red);border-top:2px dashed var(--red)"></div>SL</div>
  </div>
</div>

<!-- ═══ MAIN ══════════════════════════════════════════════════════════ -->
<div id="main">

  <div id="chart-wrap"><div id="chart"></div></div>

  <div id="vresize"></div>

  <div id="sidebar">

    <!-- Panel 1 — Transaction History -->
    <div class="panel" id="panel-tx" style="flex:1 1 50%">
      <div class="ph">
        <span>📋 Transaction History</span>
        <span class="pbadge" id="tx-count"
              style="background:rgba(0,212,170,.12);color:var(--green)">0 trades</span>
      </div>
      <div class="pbody" id="tx-body">
        <div class="empty">Waiting for trades…</div>
      </div>
    </div>

    <div id="hresize"></div>

    <!-- Panel 2 — Event / Error Log -->
    <div class="panel" id="panel-log" style="flex:1 1 50%">
      <div class="ph">
        <span>🔔 Event Log</span>
        <div style="display:flex;gap:8px;align-items:center">
          <span class="pbadge" id="err-count"
                style="background:rgba(255,77,109,.12);color:var(--red)">0E · 0W</span>
          <button class="fbtn on" id="fbtn" title="Auto-scroll">↓ Follow</button>
        </div>
      </div>
      <div class="pbody" id="log-body">
        <ul id="log-list">
          <li><div class="empty">Waiting for log data…</div></li>
        </ul>
      </div>
    </div>

  </div><!-- /sidebar -->
</div><!-- /main -->

<script>
'use strict';

/* ════ CHART ════════════════════════════════════════════════════════ */
const chart = LightweightCharts.createChart(document.getElementById('chart'), {
  layout:          { background: { color: '#0a0e14' }, textColor: '#4a5568' },
  grid:            { vertLines: { color: '#0d1a28' }, horzLines: { color: '#0d1a28' } },
  crosshair:       { mode: 0 },
  rightPriceScale: { borderColor: '#1e2d40' },
  timeScale:       { borderColor: '#1e2d40', timeVisible: true },
  localization: {
    timeFormatter: (ts) => {
      const d = new Date(ts * 1000);
      return d.toLocaleDateString([], {month:'2-digit', day:'2-digit'})
        + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', hour12: false});
    }
  }
});
const haSeries    = chart.addSeries(LightweightCharts.CandlestickSeries,
  {upColor:'#00d4aa',downColor:'#ff4d6d',borderUpColor:'#00d4aa',
   borderDownColor:'#ff4d6d',wickUpColor:'#00d4aa',wickDownColor:'#ff4d6d'});
const vwapS       = chart.addSeries(LightweightCharts.LineSeries,
  {color:'rgba(245,200,66,.85)',lineWidth:2,title:'VWAP',visible:true});
const vwapU1      = chart.addSeries(LightweightCharts.LineSeries,
  {color:'rgba(245,200,66,.40)',lineWidth:1,lineStyle:1,title:'+0.8σ',visible:true});
const vwapL1      = chart.addSeries(LightweightCharts.LineSeries,
  {color:'rgba(245,200,66,.40)',lineWidth:1,lineStyle:1,title:'-0.8σ',visible:true});
const vwapU2      = chart.addSeries(LightweightCharts.LineSeries,
  {color:'rgba(245,200,66,.18)',lineWidth:1,lineStyle:2,title:'+1.6σ',visible:true});
const vwapL2      = chart.addSeries(LightweightCharts.LineSeries,
  {color:'rgba(245,200,66,.18)',lineWidth:1,lineStyle:2,title:'-1.6σ',visible:true});
const trailS      = chart.addSeries(LightweightCharts.LineSeries,
  {color:'#ff4d6d',lineWidth:2.5,lineStyle:2,title:'Bot SL',visible:true});

let firstLoad = true, lastTime = null;

// VWAP visibility toggles
document.getElementById('vwap-toggle').addEventListener('change', function() {
  vwapS.applyOptions({visible: this.checked});
});
document.getElementById('vwap-band1-toggle').addEventListener('change', function() {
  vwapU1.applyOptions({visible: this.checked});
  vwapL1.applyOptions({visible: this.checked});
});
document.getElementById('vwap-band2-toggle').addEventListener('change', function() {
  vwapU2.applyOptions({visible: this.checked});
  vwapL2.applyOptions({visible: this.checked});
});

function resizeChart(){
  const w = document.getElementById('chart-wrap');
  chart.applyOptions({width:w.offsetWidth, height:w.offsetHeight});
}

async function pollChart(){
  try{
    const r    = await fetch('/api/live_chart_data');
    if(!r.ok) throw 0;
    const d    = await r.json();
    const ha   = d.ha;
    if(!ha?.length) return;
    const lt   = ha[ha.length-1].time;
    
    if(firstLoad || lt !== lastTime){
      haSeries.setData(ha);
      try { haSeries.setMarkers(d.markers); } catch(e) {}
      if(d.vwap && d.vwap.length > 0) vwapS.setData(d.vwap);
      if(d.vwap_upper1 && d.vwap_upper1.length > 0) vwapU1.setData(d.vwap_upper1);
      if(d.vwap_lower1 && d.vwap_lower1.length > 0) vwapL1.setData(d.vwap_lower1);
      if(d.trail_sl && d.trail_sl.length > 0) trailS.setData(d.trail_sl);
      lastTime = lt;
    } else {
      haSeries.update(ha[ha.length-1]);
      if(d.vwap && d.vwap.length > 0) vwapS.update(d.vwap[d.vwap.length-1]);
      if(d.trail_sl && d.trail_sl.length > 0) trailS.update(d.trail_sl[d.trail_sl.length-1]);
    }
    if(firstLoad){ 
      chart.timeScale().fitContent(); 
      firstLoad=false; 
    }
    document.getElementById('tb-price').textContent = '$'+d.latest_price.toFixed(1);
    document.getElementById('tb-sl').textContent    = d.latest_sl
      ? '$'+d.latest_sl.toFixed(1) : 'No Position';
    document.getElementById('tb-status').textContent =
      'Synced '+new Date().toLocaleTimeString();
  } catch {
    document.getElementById('tb-status').textContent = 'Disconnected — retrying…';
  }
}

/* ════ TRANSACTION TABLE ════════════════════════════════════════════ */
let txCount = 0;

function renderTrades(trades){
  const body  = document.getElementById('tx-body');
  const badge = document.getElementById('tx-count');
  badge.textContent = `${trades.length} trade${trades.length!==1?'s':''}`;
  if(!trades.length){
    body.innerHTML='<div class="empty">No trades recorded yet.</div>'; return;
  }
  if(trades.length === txCount) return;   // no change
  txCount = trades.length;

  const rows = trades.map(t=>{
    const buy  = t.side==='buy';
    const risk = (t.entry && t.stop) ? Math.abs(t.entry-t.stop).toFixed(1) : '—';
    const scls = {placed:'sp',filled:'sf',cancelled:'sc'}[t.status] ?? 'sc';
    return `<tr>
      <td style="color:var(--muted);font-size:10px">${t.ts_fmt}</td>
      <td class="${buy?'sl':'ss'}">${buy?'▲ LONG':'▼ SHORT'}</td>
      <td>${t.entry?t.entry.toFixed(1):'—'}</td>
      <td style="color:var(--muted)">${t.stop?t.stop.toFixed(1):'—'}</td>
      <td style="color:var(--muted)">${risk}</td>
      <td>${t.size??'—'}</td>
      <td class="${scls}">${t.status??'—'}</td>
    </tr>`;
  }).join('');

  body.innerHTML=`<table id="tx-table">
    <thead><tr>
      <th>Time</th><th>Side</th><th>Entry</th>
      <th>Stop</th><th>Risk$</th><th>Qty</th><th>Status</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

/* ════ EVENT LOG ════════════════════════════════════════════════════ */
let follow = true, logHash = '';

document.getElementById('fbtn').addEventListener('click', function(){
  follow = !follow;
  this.classList.toggle('on', follow);
  this.textContent = follow ? '↓ Follow' : '⏸ Paused';
});

const EXTRA_KEYS = ['side','entry','stop','size','old_sl','new_sl',
                    'error','loss_pct','current_price','vwap'];

function renderLog(entries){
  const body  = document.getElementById('log-body');
  const list  = document.getElementById('log-list');
  const badge = document.getElementById('err-count');

  const h = entries.length + (entries[0] ? entries[0].ts+entries[0].message : '');
  if(h === logHash) return;
  logHash = h;

  const ec = entries.filter(e=>e.level==='ERROR').length;
  const wc = entries.filter(e=>e.level==='WARNING').length;
  badge.textContent = `${ec}E · ${wc}W`;
  badge.style.color  = ec>0 ? 'var(--red)' : wc>0 ? 'var(--warn)' : 'var(--muted)';

  if(!entries.length){
    list.innerHTML='<li><div class="empty">No log entries found.<br>'
      +'<span style="font-size:10px">Check LOG_DIR env var.</span></div></li>';
    return;
  }

  const lcls = {WARNING:'lW',ERROR:'lE'};

  const items = entries.map(e=>{
    const cls   = lcls[e.level] || '';
    const extras= EXTRA_KEYS
      .filter(k=>e.extra&&e.extra[k]!=null)
      .map(k=>`<span>${k}=<b>${e.extra[k]}</b></span>`)
      .join(' ');
    const extHtml = extras ? `<div class="lext">${extras}</div>` : '';
    return `<li class="${cls}">
      <div class="lts">${e.ts}</div>
      <div class="lsec">${e.section}</div>
      <div class="lmsg">${e.message}${extHtml}</div>
    </li>`;
  }).join('');

  list.innerHTML = items;
  if(follow) body.scrollTop = 0;
}

/* ════ DASHBOARD POLL ═══════════════════════════════════════════════ */
async function pollDashboard(){
  try{
    const r = await fetch('/api/dashboard_data');
    if(!r.ok) return;
    const d = await r.json();
    renderTrades(d.trades||[]);
    renderLog(d.log||[]);
  } catch{ /* silent */ }
}

/* ════ VERTICAL RESIZE (chart ↔ sidebar) ════════════════════════════ */
(()=>{
  const h=document.getElementById('vresize');
  const sb=document.getElementById('sidebar');
  let drag=false, sx=0, sw=0;
  h.addEventListener('mousedown',e=>{
    drag=true;sx=e.clientX;sw=sb.offsetWidth;
    h.classList.add('drag');
    document.body.style.cssText='cursor:col-resize;user-select:none';
  });
  document.addEventListener('mousemove',e=>{
    if(!drag)return;
    const nw=Math.min(680,Math.max(220,sw+(sx-e.clientX)));
    sb.style.width=nw+'px'; resizeChart();
  });
  document.addEventListener('mouseup',()=>{
    if(!drag)return; drag=false;
    h.classList.remove('drag');
    document.body.style.cssText=''; resizeChart();
  });
})();

/* ════ HORIZONTAL RESIZE (tx ↔ log) ════════════════════════════════ */
(()=>{
  const h=document.getElementById('hresize');
  const pt=document.getElementById('panel-tx');
  const pl=document.getElementById('panel-log');
  const sb=document.getElementById('sidebar');
  let drag=false,sy=0,sh=0;
  h.addEventListener('mousedown',e=>{
    drag=true;sy=e.clientY;sh=pt.offsetHeight;
    h.classList.add('drag');
    document.body.style.cssText='cursor:row-resize;user-select:none';
  });
  document.addEventListener('mousemove',e=>{
    if(!drag)return;
    const tot = sb.offsetHeight - h.offsetHeight;
    const min = 80;
    const nh  = Math.min(tot-min,Math.max(min,sh+(e.clientY-sy)));
    pt.style.flex='none'; pl.style.flex='none';
    pt.style.height=nh+'px'; pl.style.height=(tot-nh)+'px';
  });
  document.addEventListener('mouseup',()=>{
    if(!drag)return; drag=false;
    h.classList.remove('drag');
    document.body.style.cssText='';
  });
})();

/* ════ BOOT ════════════════════════════════════════════════════════ */
window.addEventListener('resize', resizeChart);
requestAnimationFrame(()=>requestAnimationFrame(resizeChart));

pollChart();
pollDashboard();
setInterval(pollChart,     5000);
setInterval(pollDashboard, 5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════╗
║   BTCUSD HA Live Dashboard  —  v3            ║
║   DB  : {DB_PATH:<35}║
║   Log : {LOG_DIR:<35}║
║   URL : http://{HOST}:{PORT:<27}║
╚══════════════════════════════════════════════╝
""")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)