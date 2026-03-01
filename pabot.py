from dotenv import load_dotenv
import os
import json
import logging
import uuid
import time
import math
import hashlib
import hmac
import signal
import sys
import sqlite3
import threading
import requests
import numpy as np
import pandas as pd
import schedule
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from delta_rest_client import DeltaRestClient

# ==========================================
# CONFIG
# ==========================================

load_dotenv()

API_KEY    = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL   = os.getenv("BASE_URL", "https://api.india.delta.exchange")
PRODUCT_ID = int(os.getenv("PRODUCT_ID", 27))

RISK_PERCENT         = float(os.getenv("RISK_PERCENT", 0.01))
LEVERAGE             = int(os.getenv("LEVERAGE", 2000))
MAX_DAILY_LOSS       = float(os.getenv("MAX_DAILY_LOSS", 0.45))
LOG_DIR              = os.getenv("LOG_DIR", "logs_struct")
DB_PATH              = os.getenv("DB_PATH", "ha_struct.db")
MIN_PULLBACK_CANDLES = int(os.getenv("MIN_PULLBACK_CANDLES", 2))
VWAP_RESET_HOURS     = int(os.getenv("VWAP_RESET_HOURS", 4))
SIGNAL_LATENCY_BARS  = int(os.getenv("SIGNAL_LATENCY_BARS", 2))  # FIX-5: allow N-candle confirm latency

# ==========================================
# STARTUP VALIDATION
# ==========================================

def validate_config():
    errors = []
    if not API_KEY:    errors.append("API_KEY is not set")
    if not API_SECRET: errors.append("API_SECRET is not set")
    if not (0 < RISK_PERCENT < 0.1): errors.append("RISK_PERCENT outside safe range")
    if MIN_PULLBACK_CANDLES < 1:     errors.append("MIN_PULLBACK_CANDLES must be >= 1")
    if errors:
        for e in errors: print(f"[CONFIG ERROR] {e}")
        sys.exit(1)

validate_config()

delta = DeltaRestClient(base_url=BASE_URL, api_key=API_KEY, api_secret=API_SECRET)

# ==========================================
# LOGGING
# ==========================================

os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/struct_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": record.levelname}
        if hasattr(record, "extra_data"): log_record.update(record.extra_data)
        else: log_record["message"] = record.getMessage()
        return json.dumps(log_record, cls=SafeJSONEncoder)

logger = logging.getLogger("ha_struct_logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_filename)
fh.setFormatter(JSONFormatter())
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.handlers.clear()
logger.addHandler(fh)
logger.addHandler(ch)

def log(section, msg, key="SYSTEM", level="INFO", **kwargs):
    data = {"cycle_id": key, "section": section, "message": msg}
    data.update(kwargs)
    if level == "ERROR":   logger.error(msg,   extra={"extra_data": data})
    elif level == "WARNING": logger.warning(msg, extra={"extra_data": data})
    else:                    logger.info(msg,    extra={"extra_data": data})

def shutdown(sig, frame):
    log("SYSTEM", "Shutdown signal received — exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ==========================================
# DATABASE
# ==========================================

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                time        INTEGER PRIMARY KEY,
                open        REAL, high       REAL, low        REAL,
                close       REAL, volume     REAL,
                ha_open     REAL, ha_high    REAL, ha_low     REAL,
                ha_close    REAL, ha_green   INTEGER,
                vwap        REAL, vwap_upper1 REAL, vwap_lower1 REAL,
                vwap_upper2 REAL, vwap_lower2 REAL, std_dev    REAL,
                is_pivot    TEXT DEFAULT NULL,
                active_sl   REAL DEFAULT NULL,
                recorded_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id     TEXT, timestamp TEXT, candle_time INTEGER, side TEXT,
                entry        REAL, stop REAL, size INTEGER, status TEXT DEFAULT 'placed'
            );
        """)

def db_save_candles(rows: list):
    try:
        with get_db() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO candles
                  (time, open, high, low, close, volume,
                   ha_open, ha_high, ha_low, ha_close, ha_green,
                   vwap, vwap_upper1, vwap_lower1, vwap_upper2, vwap_lower2, std_dev,
                   is_pivot, active_sl, recorded_at)
                VALUES
                  (:time, :open, :high, :low, :close, :volume,
                   :ha_open, :ha_high, :ha_low, :ha_close, :ha_green,
                   :vwap, :vwap_upper1, :vwap_lower1, :vwap_upper2, :vwap_lower2, :std_dev,
                   :is_pivot, :active_sl, :recorded_at)
            """, rows)
    except Exception as e:
        log("DB", "candle batch save failed", level="ERROR", error=str(e))

def db_save_trade(trade: dict):
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO trades (cycle_id, timestamp, candle_time, side, entry, stop, size, status)
                VALUES (:cycle_id, :timestamp, :candle_time, :side, :entry, :stop, :size, :status)
            """, trade)
    except Exception as e:
        log("DB", "trade save failed", level="ERROR", error=str(e))

# ==========================================
# FIX-1: DAILY LOSS CIRCUIT BREAKER
# ==========================================

def get_daily_loss_pct() -> float:
    """
    Returns today's realised loss as a fraction of starting equity.
    Reads from the exchange wallet history; falls back to 0.0 on error.
    """
    try:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        data = signed_request(
            "GET",
            f"/v2/wallet/transactions?asset_symbol=USD&"
            f"start_time={int(today_start.timestamp())}&end_time={int(time.time())}"
        )
        transactions = data.get("result", [])
        # Sum up realised PnL entries (negative values = losses)
        daily_pnl = sum(float(t.get("amount", 0)) for t in transactions if t.get("transaction_type") == "realised_pnl")
        equity = get_equity()
        if equity <= 0:
            return 0.0
        loss_pct = -daily_pnl / (equity + abs(min(daily_pnl, 0)))  # loss as fraction of peak
        return max(loss_pct, 0.0)
    except Exception as e:
        log("RISK", "Daily loss calculation failed — defaulting to 0", level="WARNING", error=str(e))
        return 0.0

def is_daily_loss_exceeded() -> bool:
    loss = get_daily_loss_pct()
    if loss >= MAX_DAILY_LOSS:
        log("RISK", "DAILY LOSS LIMIT REACHED — halting new entries",
            level="WARNING", loss_pct=round(loss * 100, 2), limit_pct=round(MAX_DAILY_LOSS * 100, 2))
        return True
    return False

# ==========================================
# API UTILS
# ==========================================

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def signed_request(method, path, payload=None):
    timestamp = str(int(time.time()))
    body = json.dumps(payload) if payload else ""
    sig_data = method + timestamp + path + body
    signature = hmac.new(API_SECRET.encode(), sig_data.encode(), hashlib.sha256).hexdigest()
    headers = {
        "api-key": API_KEY, "timestamp": timestamp,
        "signature": signature, "Content-Type": "application/json"
    }
    response = requests.request(method, BASE_URL + path, headers=headers, data=body, timeout=10)
    if not response.ok:
        log("API", "Request failed", level="ERROR",
            status_code=response.status_code, response=response.text[:500])
        response.raise_for_status()
    return response.json()

def get_equity() -> float:
    try:
        data = signed_request("GET", "/v2/wallet/balances")
        for asset in data.get("result", []):
            if asset.get("asset_symbol") == "USD":
                val = float(asset["available_balance"])
                if val <= 0:
                    log("API", "get_equity returned 0 — possible API issue", level="WARNING")
                return val
        return 0.0
    except Exception as e:
        log("API", "get_equity failed", level="WARNING", error=str(e))
        return 0.0

def get_position():
    # FIX-2: named exception, logged properly
    try:
        pos = delta.get_position(PRODUCT_ID)
        if not pos: return None
        data = pos.get("result") or pos.get("data") or pos
        size = float(data.get("size", 0))
        if size != 0:
            return {"size": size, "entry": float(data.get("entry_price", 0))}
        return None
    except Exception as e:
        log("API", "get_position failed", level="WARNING", error=str(e))
        return None

# ==========================================
# FIX-6: SINGLE CONSOLIDATED ORDER FETCH
# ==========================================

def get_all_open_orders() -> tuple[bool, object]:
    """
    Single API call that returns (has_limit_orders: bool, stop_order: dict | None).
    Replaces the three separate calls: has_open_entry_orders(), get_open_stop_order().
    """
    try:
        data = signed_request("GET", f"/v2/orders?product_id={PRODUCT_ID}&state=open")
        orders = data.get("result", [])
        # AFTER — exclude any limit order that has a stop_price set (those are bracket SL children)
        limit_orders = [
            o for o in orders
            if o["order_type"] == "limit_order"
            and not o.get("stop_price")          # bracket SL children have stop_price populated
            and not o.get("bracket_order")       # or are flagged as bracket children
        ]
        stop_orders  = [o for o in orders if o["order_type"] in ("stop_order", "stop_limit_order")]
        has_limit    = len(limit_orders) > 0
        stop_order   = stop_orders[0] if stop_orders else None
        return has_limit, stop_order, orders
    except Exception as e:
        log("API", "get_all_open_orders failed — assuming limit exists for safety",
            level="WARNING", error=str(e))
        return True, None, []   # safe default: block new entries

# ==========================================
# DATA & INDICATORS
# ==========================================

def get_candles(minutes: int = 400) -> pd.DataFrame | None:
    # FIX-2: named exception, logged properly
    try:
        now = datetime.now(timezone.utc)
        params = {
            "symbol": "BTCUSD", "resolution": "1m",
            "start": int((now - timedelta(minutes=minutes)).timestamp()),
            "end":   int(now.timestamp())
        }
        r = requests.get(f"{BASE_URL}/v2/history/candles", params=params, timeout=10)
        data = r.json().get("result")
        if not data: return None
        df = (pd.DataFrame(data)
              .rename(columns={"t": "time", "o": "open", "h": "high",
                               "l": "low",  "c": "close", "v": "volume"})
              .astype(float))
        df["time"] = df["time"].astype(int)
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        log("DATA", "get_candles failed", level="WARNING", error=str(e))
        return None

def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    # Use numpy array directly — avoids repeated .iloc[] overhead
    ha_close_vals = ha["ha_close"].values
    ha_open = np.empty(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close_vals[i - 1]) * 0.5
    ha["ha_open"]  = ha_open
    ha["ha_high"]  = ha[["high",  "ha_open", "ha_close"]].max(axis=1)
    ha["ha_low"]   = ha[["low",   "ha_open", "ha_close"]].min(axis=1)
    ha["ha_green"] = ha["ha_close"] > ha["ha_open"]
    return ha

def get_4h_session_start(now: datetime) -> datetime:
    hour_block = (now.hour // VWAP_RESET_HOURS) * VWAP_RESET_HOURS
    return now.replace(hour=hour_block, minute=0, second=0, microsecond=0)

def compute_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    now        = datetime.now(timezone.utc)
    session_ts = int(get_4h_session_start(now).timestamp())
    vwap_cols  = ["vwap", "vwap_upper1", "vwap_lower1", "vwap_upper2", "vwap_lower2", "std_dev"]

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]        = df["typical_price"] * df["volume"]
    df["in_session"]    = df["time"] >= session_ts
    session_df          = df[df["in_session"]].copy()

    if len(session_df) < 2:
        for col in vwap_cols: df[col] = np.nan
        return df

    session_df["cum_tp_vol"] = session_df["tp_vol"].cumsum()
    session_df["cum_vol"]    = session_df["volume"].cumsum()
    session_df["vwap"]       = session_df["cum_tp_vol"] / session_df["cum_vol"]

    # FIX-3: Volume-weighted std dev (matches TradingView anchored VWAP bands)
    sq_dev_vol                = ((session_df["typical_price"] - session_df["vwap"]) ** 2
                                  * session_df["volume"])
    session_df["cum_sq_dev"] = sq_dev_vol.cumsum()
    session_df["std_dev"]    = np.sqrt(session_df["cum_sq_dev"] / session_df["cum_vol"])

    session_df["vwap_upper1"] = session_df["vwap"] + 0.8 * session_df["std_dev"]
    session_df["vwap_lower1"] = session_df["vwap"] - 0.8 * session_df["std_dev"]
    session_df["vwap_upper2"] = session_df["vwap"] + 1.6 * session_df["std_dev"]
    session_df["vwap_lower2"] = session_df["vwap"] - 1.6 * session_df["std_dev"]

    for col in vwap_cols:
        df[col] = np.nan
        df.loc[df["in_session"], col] = session_df[col].values
    return df

def get_market_structure(df: pd.DataFrame, min_candles: int):
    swing_lows, swing_highs = [], []
    current_color, streak_start = None, 0

    for i in range(len(df)):
        color = "green" if bool(df.iloc[i]["ha_green"]) else "red"

        if current_color is None:
            current_color, streak_start = color, i
            continue

        if color != current_color:
            if (i - streak_start) >= min_candles:
                streak_df = df.iloc[streak_start:i]
                if current_color == "red":
                    min_idx = streak_df["low"].idxmin()
                    swing_lows.append({
                        "price":       float(df.iloc[min_idx]["low"]),
                        "pivot_idx":   int(min_idx),
                        "confirm_idx": i
                    })
                else:
                    max_idx = streak_df["high"].idxmax()
                    swing_highs.append({
                        "price":       float(df.iloc[max_idx]["high"]),
                        "pivot_idx":   int(max_idx),
                        "confirm_idx": i
                    })
            current_color, streak_start = color, i

    return swing_lows, swing_highs

def add_pivot_labels(df: pd.DataFrame, swing_lows: list, swing_highs: list):
    df["is_pivot"] = None
    for i, L in enumerate(swing_lows):
        label = "HL" if i > 0 and L["price"] > swing_lows[i - 1]["price"] else "LL"
        df.at[L["pivot_idx"], "is_pivot"] = label
    for i, H in enumerate(swing_highs):
        label = "LH" if i > 0 and H["price"] < swing_highs[i - 1]["price"] else "HH"
        df.at[H["pivot_idx"], "is_pivot"] = label
    return df

# ==========================================
# EXECUTION & TRAILING
# ==========================================

def execute_entry(side: str, entry: float, stop: float, size: int, key: str):
    if size <= 0: return
    sl_limit = stop * 0.998 if side == "buy" else stop * 1.002
    try:
        log("TRADE", "Placing MARKET entry + SL bracket", key,
            side=side, size=size, expected_entry=round(entry, 1), stop=round(stop, 1))
        response = signed_request("POST", "/v2/orders", payload={
            "product_id":                    PRODUCT_ID,
            "size":                          size,
            "side":                          side,
            "order_type":                    "market_order",
            "bracket_stop_loss_price":       str(round(stop, 1)),
            "bracket_stop_loss_limit_price": str(round(sl_limit, 1))
        })
        log("ORDER", "Market Entry placed", key, response=response)
    except Exception as e:
        log("TRADE", "Entry execution failed", key, level="ERROR", error=str(e))

def update_trailing_sl(order_id: int, new_sl: float, side: str, key: str):
    sl_limit = new_sl * 0.998 if side == "buy" else new_sl * 1.002
    try:
        log("TRAIL", f"Updating SL to {round(new_sl, 1)}", key)
        signed_request("PUT", "/v2/orders", payload={
            "id":          order_id,
            "product_id":  PRODUCT_ID,
            "stop_price":  str(round(new_sl, 1)),
            "limit_price": str(round(sl_limit, 1))
        })
        log("TRAIL", "SL updated successfully", key)
    except Exception as e:
        log("TRAIL", "SL update failed", key, level="ERROR", error=str(e))

# ==========================================
# MAIN STRATEGY LOOP
# ==========================================

last_processed_time = None
# FIX-7: thread-safe lock instead of bare boolean
_trade_lock = threading.Lock()

def run_strategy():
    global last_processed_time

    # Bail immediately if the lock is held (another cycle is executing)
    if not _trade_lock.acquire(blocking=False):
        return

    try:
        cycle_id = str(uuid.uuid4())[:8]

        # FIX-1: Enforce daily loss circuit breaker before any work
        if is_daily_loss_exceeded():
            return

        # 1. Fetch & Prep Data
        df = get_candles()
        if df is None or len(df) < 60:
            log("STRATEGY", "Failed to fetch candles or insufficient data", cycle_id, level="WARNING")
            return

        now               = datetime.now(timezone.utc)
        current_minute_ts = int(now.replace(second=0, microsecond=0).timestamp())
        closed_df         = df[df["time"] < current_minute_ts]
        if len(closed_df) < 60: return

        signal_idx  = closed_df.index[-1]
        candle_time = int(df.iloc[signal_idx]["time"])
        raw_close   = float(df.iloc[signal_idx]["close"])

        if last_processed_time == candle_time: return
        last_processed_time = candle_time

        df = compute_heikin_ashi(df)
        df = compute_vwap_bands(df)
        swing_lows, swing_highs = get_market_structure(df, MIN_PULLBACK_CANDLES)
        df = add_pivot_labels(df, swing_lows, swing_highs)

        # FIX-4: Compute VWAP at signal candle for trend filter
        vwap_at_signal = None
        vwap_val = df.iloc[signal_idx]["vwap"]
        if not pd.isna(vwap_val):
            vwap_at_signal = float(vwap_val)

        # 2. Check Open Position & Resting Orders — FIX-6: single API call
        position                  = get_position()
        open_entry_exists, sl_order, orders = get_all_open_orders()
        active_sl_val             = None

        if open_entry_exists:
            log("STRATEGY", "Resting entry order exists — waiting", cycle_id,
                open_order_types=[o["order_type"] for o in orders],
                open_order_stops=[o.get("stop_price") for o in orders])

        elif position is not None:
            # IN A TRADE — manage trailing SL
            pos_size = position["size"]
            pos_side = "buy" if pos_size > 0 else "sell"

            if sl_order:
                current_sl = float(sl_order.get("stop_price", 0))
                
                # FIX-8: Load last known active_sl from DB so it persists across cycles
                # If no new swing is detected, this ensures the SL doesn't get overwritten with NULL
                try:
                    with get_db() as conn:
                        row = conn.execute(
                            "SELECT active_sl FROM candles WHERE active_sl IS NOT NULL ORDER BY time DESC LIMIT 1"
                        ).fetchone()
                        active_sl_val = float(row[0]) if row and row[0] else current_sl
                except Exception as e:
                    log("DB", "Failed to load last active_sl", cycle_id, level="WARNING", error=str(e))
                    active_sl_val = current_sl

                if pos_side == "buy" and swing_lows:
                    last_L = swing_lows[-1]
                    if last_L["price"] > current_sl:
                        log("STRATEGY", "New Swing Low — Trailing SL Up", cycle_id,
                            old_sl=current_sl, new_sl=last_L["price"])
                        update_trailing_sl(sl_order["id"], last_L["price"], pos_side, cycle_id)
                        active_sl_val = last_L["price"]

                elif pos_side == "sell" and swing_highs:
                    last_H = swing_highs[-1]
                    if last_H["price"] < current_sl:
                        log("STRATEGY", "New Swing High — Trailing SL Down", cycle_id,
                            old_sl=current_sl, new_sl=last_H["price"])
                        update_trailing_sl(sl_order["id"], last_H["price"], pos_side, cycle_id)
                        active_sl_val = last_H["price"]

        else:
            # NO POSITION — look for entry

            # LONG SETUP: First Higher Low
            if len(swing_lows) >= 3:
                L0, L1, L2 = swing_lows[-1], swing_lows[-2], swing_lows[-3]

                # FIX-5: allow up to SIGNAL_LATENCY_BARS candles of latency
                confirm_fresh = abs(L0["confirm_idx"] - signal_idx) <= SIGNAL_LATENCY_BARS

                # FIX-4: price must be above VWAP for a long entry
                #above_vwap = vwap_at_signal is None or raw_close > vwap_at_signal
                if confirm_fresh:
                #if confirm_fresh and above_vwap:
                    if L1["price"] < L2["price"] and L0["price"] > L1["price"]:
                        log("STRATEGY", "LONG Signal: First Higher Low Detected", cycle_id,
                            L2=L2["price"], L1=L1["price"], L0=L0["price"],
                            vwap_filter=vwap_at_signal)

                        entry  = raw_close
                        stop   = L1["price"]
                        equity = get_equity()
                        if equity <= 0:
                            log("STRATEGY", "Skipping entry — equity is 0, API may be unavailable", 
                                cycle_id, level="WARNING")
                            # skip but don't mark candle as processed so it retries next cycle
                            last_processed_time = None
                            return
                        # FIX-7b: math.floor — never risk more than RISK_PERCENT
                        size   = math.floor((equity * RISK_PERCENT * LEVERAGE) / abs(entry - stop)) if abs(entry - stop) > 0 else 0

                        if stop < entry and size > 0:
                            execute_entry("buy", entry, stop, size, cycle_id)
                            active_sl_val = stop
                            db_save_trade({
                                "cycle_id": cycle_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                                "candle_time": candle_time, "side": "buy",
                                "entry": entry, "stop": stop, "size": size, "status": "placed"
                            })
                # elif not above_vwap:
                #     log("STRATEGY", "LONG skipped — price below VWAP", cycle_id,
                #         price=raw_close, vwap=vwap_at_signal)

            # SHORT SETUP: First Lower High
            if len(swing_highs) >= 3:
                H0, H1, H2 = swing_highs[-1], swing_highs[-2], swing_highs[-3]

                confirm_fresh = abs(H0["confirm_idx"] - signal_idx) <= SIGNAL_LATENCY_BARS

                # FIX-4: price must be below VWAP for a short entry
                #below_vwap = vwap_at_signal is None or raw_close < vwap_at_signal

                if confirm_fresh: # and below_vwap:
                    if H1["price"] > H2["price"] and H0["price"] < H1["price"]:
                        log("STRATEGY", "SHORT Signal: First Lower High Detected", cycle_id,
                            H2=H2["price"], H1=H1["price"], H0=H0["price"],
                            vwap_filter=vwap_at_signal)

                        entry  = raw_close
                        stop   = H1["price"]
                        equity = get_equity()
                        if equity <= 0:
                            log("STRATEGY", "Skipping entry — equity is 0, API may be unavailable", 
                                cycle_id, level="WARNING")
                            # skip but don't mark candle as processed so it retries next cycle
                            last_processed_time = None
                            return
                        size   = math.floor((equity * RISK_PERCENT * LEVERAGE) / abs(entry - stop)) if abs(entry - stop) > 0 else 0

                        log("STRATEGY", "SHORT size calculation", cycle_id,
                            equity=equity, risk_pct=RISK_PERCENT, leverage=LEVERAGE,
                            entry=entry, stop=stop, risk_pts=abs(entry-stop), size=size)
                        if stop > entry and size > 0:
                            execute_entry("sell", entry, stop, size, cycle_id)
                            active_sl_val = stop
                            db_save_trade({
                                "cycle_id": cycle_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                                "candle_time": candle_time, "side": "sell",
                                "entry": entry, "stop": stop, "size": size, "status": "placed"
                            })
                        else:
                            log("STRATEGY", "SHORT blocked — size=0 or invalid stop", cycle_id,
                                size=size, stop=stop, entry=entry, equity=equity, level="WARNING")
                # elif not below_vwap:
                #     log("STRATEGY", "SHORT skipped — price above VWAP", cycle_id,
                #         price=raw_close, vwap=vwap_at_signal)

        # 3. Save to Database
        db_rows    = []
        recent_df  = df.iloc[-20:].copy()
        for _, row in recent_df.iterrows():
            db_rows.append({
                "time":        int(row["time"]),
                "open":        float(row["open"]),   "high":  float(row["high"]),
                "low":         float(row["low"]),    "close": float(row["close"]),
                "volume":      float(row["volume"]),
                "ha_open":     float(row["ha_open"]), "ha_high": float(row["ha_high"]),
                "ha_low":      float(row["ha_low"]),  "ha_close": float(row["ha_close"]),
                "ha_green":    int(row["ha_green"]),
                "vwap":        float(row["vwap"])        if not pd.isna(row["vwap"])        else None,
                "vwap_upper1": float(row["vwap_upper1"]) if not pd.isna(row["vwap_upper1"]) else None,
                "vwap_lower1": float(row["vwap_lower1"]) if not pd.isna(row["vwap_lower1"]) else None,
                "vwap_upper2": float(row["vwap_upper2"]) if not pd.isna(row["vwap_upper2"]) else None,
                "vwap_lower2": float(row["vwap_lower2"]) if not pd.isna(row["vwap_lower2"]) else None,
                "std_dev":     float(row["std_dev"])     if not pd.isna(row["std_dev"])     else None,
                "is_pivot":    row["is_pivot"] if pd.notna(row["is_pivot"]) else None,
                "active_sl":   float(active_sl_val) if active_sl_val else None,
                "recorded_at": datetime.now(timezone.utc).isoformat()
            })
        db_save_candles(db_rows)

        log("STRATEGY", "Cycle complete", cycle_id,
            current_price=raw_close,
            swing_lows_found=len(swing_lows),
            swing_highs_found=len(swing_highs),
            vwap=vwap_at_signal)

    finally:
        _trade_lock.release()

# ==========================================
# BOOT
# ==========================================

schedule.every(10).seconds.do(run_strategy)

init_db()
log("SYSTEM", "Dow Theory HA Strategy — BTCUSD 1m Started")

run_strategy()

while True:
    schedule.run_pending()
    time.sleep(1)