import argparse
import json
import math
import ssl
import statistics
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


FAPI_BASE = "https://fapi.binance.com"
DEFAULT_SYMBOL = "ETHUSDT"
DEFAULT_MARKET_TYPE = "usd_m_futures"
MS_IN_HOUR = 60 * 60 * 1000
MS_IN_DAY = 24 * MS_IN_HOUR


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def iso_to_ms(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def http_get_json(path: str, params: Dict[str, object]) -> object:
    query = urllib.parse.urlencode(params)
    url = f"{FAPI_BASE}{path}?{query}"
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch Binance data after retries: {last_error}")


def chunked_kline_fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List[object]]:
    cursor = start_ms
    rows: List[List[object]] = []
    while cursor < end_ms:
        batch = http_get_json(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500},
        )
        if not batch:
            break
        rows.extend(batch)
        next_open = int(batch[-1][0]) + 1
        if next_open <= cursor:
            break
        cursor = next_open
        time.sleep(0.08)
    return rows


def chunked_funding_fetch(symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
    cursor = start_ms
    rows: List[Dict[str, object]] = []
    while cursor < end_ms:
        batch = http_get_json(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        next_time = int(batch[-1]["fundingTime"]) + 1
        if next_time <= cursor:
            break
        cursor = next_time
        time.sleep(0.08)
    return rows


def chunked_open_interest_fetch(symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
    cursor = start_ms
    rows: List[Dict[str, object]] = []
    while cursor < end_ms:
        batch = http_get_json(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": "1h", "startTime": cursor, "endTime": end_ms, "limit": 500},
        )
        if not batch:
            break
        rows.extend(batch)
        next_time = int(batch[-1]["timestamp"]) + 1
        if next_time <= cursor:
            break
        cursor = next_time
        time.sleep(0.08)
    return rows


def rows_to_arrays(rows: Sequence[Sequence[object]]) -> Dict[str, np.ndarray]:
    open_time = np.array([int(row[0]) for row in rows], dtype=np.int64)
    open_price = np.array([float(row[1]) for row in rows], dtype=np.float64)
    high_price = np.array([float(row[2]) for row in rows], dtype=np.float64)
    low_price = np.array([float(row[3]) for row in rows], dtype=np.float64)
    close_price = np.array([float(row[4]) for row in rows], dtype=np.float64)
    volume = np.array([float(row[5]) for row in rows], dtype=np.float64)
    quote_volume = np.array([float(row[7]) for row in rows], dtype=np.float64)
    return {
        "time": open_time,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "quote_volume": quote_volume,
    }


def validate_data(arrays: Dict[str, np.ndarray], interval_ms: int) -> Dict[str, object]:
    time_array = arrays["time"]
    diffs = np.diff(time_array)
    duplicate_count = int(np.sum(diffs == 0))
    missing_count = int(np.sum(diffs > interval_ms))
    anomaly_count = int(np.sum(arrays["low"] > arrays["high"]))
    return {
        "rows": int(len(time_array)),
        "duplicates": duplicate_count,
        "gaps": missing_count,
        "negative_spread_rows": anomaly_count,
        "start": ms_to_iso(int(time_array[0])) if len(time_array) else None,
        "end": ms_to_iso(int(time_array[-1])) if len(time_array) else None,
    }


def ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < period:
        return result
    alpha = 2.0 / (period + 1)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        if np.all(~np.isnan(window)):
            result[index] = np.mean(window)
            start_index = index + 1
            break
    else:
        return result
    for index in range(start_index, len(values)):
        if math.isnan(values[index]):
            result[index] = result[index - 1]
        else:
            result[index] = alpha * values[index] + (1 - alpha) * result[index - 1]
    return result


def sma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < period:
        return result
    cumsum = np.cumsum(values, dtype=np.float64)
    result[period - 1] = cumsum[period - 1] / period
    result[period:] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < period:
        return result
    for index in range(period - 1, len(values)):
        result[index] = np.std(values[index - period + 1 : index + 1])
    return result


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.full(close.shape, np.nan, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for index in range(1, len(close)):
        tr[index] = max(high[index] - low[index], abs(high[index] - close[index - 1]), abs(low[index] - close[index - 1]))
    return ema(tr, period)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for index in range(1, len(close)):
        up_move = high[index] - high[index - 1]
        down_move = low[index - 1] - low[index]
        plus_dm[index] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[index] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[index] = max(high[index] - low[index], abs(high[index] - close[index - 1]), abs(low[index] - close[index - 1]))
    atr_values = ema(tr, period)
    plus_di = 100 * ema(plus_dm, period) / np.where(atr_values == 0, np.nan, atr_values)
    minus_di = 100 * ema(minus_dm, period) / np.where(atr_values == 0, np.nan, atr_values)
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, np.nan, plus_di + minus_di)
    return ema(dx, period)


def rolling_drawdown(close: np.ndarray, period: int = 72) -> np.ndarray:
    drawdown = np.full(close.shape, np.nan, dtype=np.float64)
    for index in range(period - 1, len(close)):
        window = close[index - period + 1 : index + 1]
        peak = np.max(window)
        drawdown[index] = (window[-1] / peak) - 1.0 if peak else np.nan
    return drawdown


def swing_bias(high: np.ndarray, low: np.ndarray, lookback: int = 24) -> np.ndarray:
    bias = np.full(high.shape, np.nan, dtype=np.float64)
    for index in range(lookback, len(high)):
        recent_high = np.max(high[index - lookback : index])
        recent_low = np.min(low[index - lookback : index])
        if high[index] > recent_high and low[index] > recent_low:
            bias[index] = 1.0
        elif low[index] < recent_low and high[index] < recent_high:
            bias[index] = -1.0
        else:
            bias[index] = 0.0
    return bias


def forward_fill(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    last = np.nan
    for index, value in enumerate(result):
        if math.isnan(value):
            result[index] = last
        else:
            last = value
    return result


def funding_to_hourly(time_array: np.ndarray, funding_rows: Sequence[Dict[str, object]]) -> np.ndarray:
    hourly = np.zeros(time_array.shape, dtype=np.float64)
    if not funding_rows:
        return hourly
    funding_times = np.array([int(item["fundingTime"]) for item in funding_rows], dtype=np.int64)
    funding_rates = np.array([float(item["fundingRate"]) for item in funding_rows], dtype=np.float64)
    pointer = 0
    last_rate = 0.0
    for index, ts in enumerate(time_array):
        while pointer < len(funding_times) and funding_times[pointer] <= ts:
            last_rate = funding_rates[pointer]
            pointer += 1
        hourly[index] = last_rate / 8.0
    return hourly


def open_interest_to_hourly(time_array: np.ndarray, open_interest_rows: Sequence[Dict[str, object]]) -> np.ndarray:
    hourly = np.zeros(time_array.shape, dtype=np.float64)
    if not open_interest_rows:
        return hourly
    oi_times = np.array([int(item["timestamp"]) for item in open_interest_rows], dtype=np.int64)
    oi_values = np.array(
        [float(item.get("sumOpenInterestValue") or item.get("sumOpenInterest") or 0.0) for item in open_interest_rows],
        dtype=np.float64,
    )
    pointer = 0
    last_value = 0.0
    for index, ts in enumerate(time_array):
        while pointer < len(oi_times) and oi_times[pointer] <= ts:
            last_value = oi_values[pointer]
            pointer += 1
        hourly[index] = last_value
    return hourly


def aggregate_bars(base: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
    count = len(base["time"]) // step
    if count == 0:
        raise ValueError("not enough bars to aggregate")
    sliced = slice(len(base["time"]) - count * step, len(base["time"]))
    result: Dict[str, np.ndarray] = {}
    time_block = base["time"][sliced].reshape(count, step)
    result["time"] = time_block[:, 0]
    result["open"] = base["open"][sliced].reshape(count, step)[:, 0]
    result["high"] = np.max(base["high"][sliced].reshape(count, step), axis=1)
    result["low"] = np.min(base["low"][sliced].reshape(count, step), axis=1)
    result["close"] = base["close"][sliced].reshape(count, step)[:, -1]
    result["volume"] = np.sum(base["volume"][sliced].reshape(count, step), axis=1)
    result["quote_volume"] = np.sum(base["quote_volume"][sliced].reshape(count, step), axis=1)
    return result


def align_feature(target_time: np.ndarray, source_time: np.ndarray, source_value: np.ndarray) -> np.ndarray:
    result = np.full(target_time.shape, np.nan, dtype=np.float64)
    pointer = 0
    last = np.nan
    for index, ts in enumerate(target_time):
        while pointer < len(source_time) and source_time[pointer] <= ts:
            last = source_value[pointer]
            pointer += 1
        result[index] = last
    return result


def normalize_signal(values: np.ndarray, clip: float = 3.0) -> np.ndarray:
    clean = values.copy()
    mask = ~np.isnan(clean)
    if not np.any(mask):
        return np.zeros_like(clean)
    mean_value = np.nanmean(clean)
    std_value = np.nanstd(clean)
    if not std_value:
        return np.zeros_like(clean)
    normalized = (clean - mean_value) / std_value
    normalized = np.clip(normalized, -clip, clip)
    normalized[np.isnan(normalized)] = 0.0
    return normalized / clip


def finite_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    clean = values[~np.isnan(values)]
    if not len(clean):
        return fallback
    return float(np.percentile(clean, percentile))


@dataclass
class ResearchConfig:
    symbol: str
    market_type: str
    start_ms: int
    end_ms: int
    fee_rate: float
    slippage_rate: float
    leverage: int


def build_features(
    hourly: Dict[str, np.ndarray],
    funding_rows: Sequence[Dict[str, object]],
    open_interest_rows: Sequence[Dict[str, object]],
) -> Dict[str, np.ndarray]:
    bars_4h = aggregate_bars(hourly, 4)
    bars_1d = aggregate_bars(hourly, 24)

    ema_1h_24 = ema(hourly["close"], 24)
    ema_1h_72 = ema(hourly["close"], 72)
    sma_1h_24 = sma(hourly["close"], 24)
    adx_1h = adx(hourly["high"], hourly["low"], hourly["close"], 14)
    atr_1h = atr(hourly["high"], hourly["low"], hourly["close"], 14)
    std_1h = rolling_std(np.diff(np.log(hourly["close"]), prepend=np.nan), 24)
    drawdown_1h = rolling_drawdown(hourly["close"], 72)
    volume_sma = sma(hourly["quote_volume"], 24)
    volume_ratio = np.where(volume_sma == 0, np.nan, hourly["quote_volume"] / volume_sma)
    swing_1h = swing_bias(hourly["high"], hourly["low"], 24)

    ema_4h_21 = ema(bars_4h["close"], 21)
    ema_4h_55 = ema(bars_4h["close"], 55)
    adx_4h = adx(bars_4h["high"], bars_4h["low"], bars_4h["close"], 14)

    ema_1d_20 = ema(bars_1d["close"], 20)
    ema_1d_50 = ema(bars_1d["close"], 50)
    adx_1d = adx(bars_1d["high"], bars_1d["low"], bars_1d["close"], 14)

    slope_1h = np.where(ema_1h_24 == 0, np.nan, (ema_1h_24 - np.roll(ema_1h_24, 6)) / ema_1h_24)
    slope_4h = np.where(ema_4h_21 == 0, np.nan, (ema_4h_21 - np.roll(ema_4h_21, 3)) / ema_4h_21)
    slope_1d = np.where(ema_1d_20 == 0, np.nan, (ema_1d_20 - np.roll(ema_1d_20, 3)) / ema_1d_20)

    aligned_4h_slope = align_feature(hourly["time"], bars_4h["time"], slope_4h)
    aligned_1d_slope = align_feature(hourly["time"], bars_1d["time"], slope_1d)
    aligned_4h_adx = align_feature(hourly["time"], bars_4h["time"], adx_4h)
    aligned_1d_adx = align_feature(hourly["time"], bars_1d["time"], adx_1d)
    aligned_ema_4h_21 = align_feature(hourly["time"], bars_4h["time"], ema_4h_21)
    aligned_ema_4h_55 = align_feature(hourly["time"], bars_4h["time"], ema_4h_55)
    aligned_ema_1d_20 = align_feature(hourly["time"], bars_1d["time"], ema_1d_20)
    aligned_ema_1d_50 = align_feature(hourly["time"], bars_1d["time"], ema_1d_50)

    funding_hourly = funding_to_hourly(hourly["time"], funding_rows)
    open_interest_hourly = open_interest_to_hourly(hourly["time"], open_interest_rows)
    oi_change = np.full(hourly["close"].shape, np.nan, dtype=np.float64)
    previous_oi = np.where(open_interest_hourly[:-1] == 0, np.nan, open_interest_hourly[:-1])
    oi_change[1:] = open_interest_hourly[1:] / previous_oi - 1.0
    oi_change_ema = ema(np.nan_to_num(oi_change, nan=0.0), 6)
    funding_positive_pressure = np.maximum(funding_hourly, 0.0)
    funding_negative_pressure = np.maximum(-funding_hourly, 0.0)
    activity_spike = volume_ratio
    activity_surge = np.maximum(activity_spike - 2.4, 0.0)

    trend_score = (
        0.22 * normalize_signal(slope_1h)
        + 0.18 * normalize_signal(aligned_4h_slope)
        + 0.18 * normalize_signal(aligned_1d_slope)
        + 0.12 * normalize_signal(adx_1h - 20.0)
        + 0.08 * normalize_signal(aligned_4h_adx - 20.0)
        + 0.08 * normalize_signal(aligned_1d_adx - 20.0)
        + 0.06 * normalize_signal(drawdown_1h)
        + 0.04 * normalize_signal(volume_ratio - 1.0)
        + 0.04 * np.nan_to_num(swing_1h, nan=0.0)
    )

    long_score = (
        trend_score
        + 0.08 * normalize_signal(oi_change_ema)
        + 0.05 * normalize_signal(activity_spike - 1.0)
        - 0.07 * normalize_signal(funding_positive_pressure)
        - 0.04 * normalize_signal(activity_surge)
    )
    short_score = (
        -trend_score
        + 0.08 * normalize_signal(oi_change_ema)
        + 0.05 * normalize_signal(activity_spike - 1.0)
        - 0.07 * normalize_signal(funding_negative_pressure)
        - 0.04 * normalize_signal(activity_surge)
    )

    high_vol_cut = finite_percentile(std_1h, 85, 0.02)
    activity_burst_cut = finite_percentile(activity_spike, 92, 2.5)
    funding_extreme_cut = finite_percentile(np.abs(funding_hourly), 90, 0.00006)
    is_high_vol = (atr_1h / hourly["close"] > 0.035) | (std_1h > high_vol_cut) | (activity_spike > activity_burst_cut)
    ema_stack_up = (hourly["close"] > ema_1h_24) & (ema_1h_24 > ema_1h_72) & (aligned_ema_4h_21 > aligned_ema_4h_55) & (aligned_ema_1d_20 > aligned_ema_1d_50)
    ema_stack_down = (hourly["close"] < ema_1h_24) & (ema_1h_24 < ema_1h_72) & (aligned_ema_4h_21 < aligned_ema_4h_55) & (aligned_ema_1d_20 < aligned_ema_1d_50)

    return {
        "ema_1h_24": ema_1h_24,
        "ema_1h_72": ema_1h_72,
        "adx_1h": adx_1h,
        "atr_1h": atr_1h,
        "std_1h": std_1h,
        "drawdown_1h": drawdown_1h,
        "volume_ratio": volume_ratio,
        "funding_hourly": funding_hourly,
        "open_interest_hourly": open_interest_hourly,
        "oi_change": oi_change,
        "oi_change_ema": oi_change_ema,
        "trend_score": trend_score,
        "long_score": long_score,
        "short_score": short_score,
        "is_high_vol": is_high_vol.astype(np.int8),
        "ema_stack_up": ema_stack_up.astype(np.int8),
        "ema_stack_down": ema_stack_down.astype(np.int8),
        "aligned_4h_slope": aligned_4h_slope,
        "aligned_1d_slope": aligned_1d_slope,
        "aligned_4h_adx": aligned_4h_adx,
        "aligned_1d_adx": aligned_1d_adx,
        "swing_1h": np.nan_to_num(swing_1h, nan=0.0),
        "activity_spike": activity_spike,
        "activity_burst_cut": np.full(hourly["close"].shape, activity_burst_cut, dtype=np.float64),
        "funding_extreme_cut": np.full(hourly["close"].shape, funding_extreme_cut, dtype=np.float64),
    }


def classify_regimes(features: Dict[str, np.ndarray], params: Dict[str, object]) -> Dict[str, np.ndarray]:
    high_vol = features["is_high_vol"] == 1
    activity = np.nan_to_num(features["activity_spike"], nan=0.0)
    oi_change = np.nan_to_num(features["oi_change_ema"], nan=0.0)
    funding = features["funding_hourly"]

    long_entry = (
        (features["long_score"] >= float(params["trend_entry_threshold_long"]))
        & (features["ema_stack_up"] == 1)
        & (features["adx_1h"] >= float(params["adx_min_long"]))
        & (activity >= float(params["activity_min_long"]))
        & (activity <= float(params["activity_spike_max_long"]))
        & (oi_change >= float(params["oi_change_min_long"]))
        & (funding <= float(params["funding_extreme_abs_max_long"]))
        & ~high_vol
    )
    short_entry = (
        (features["short_score"] >= float(params["trend_entry_threshold_short"]))
        & (features["ema_stack_down"] == 1)
        & (features["adx_1h"] >= float(params["adx_min_short"]))
        & (activity >= float(params["activity_min_short"]))
        & (activity <= float(params["activity_spike_max_short"]))
        & (oi_change >= float(params["oi_change_min_short"]))
        & (funding >= -float(params["funding_extreme_abs_max_short"]))
        & ~high_vol
    )
    long_hold = (
        (features["long_score"] >= float(params["trend_exit_threshold_long"]))
        & (features["ema_stack_up"] == 1)
        & (funding <= float(params["funding_extreme_abs_max_long"]) * 1.35)
        & (activity <= float(params["activity_spike_max_long"]) * 1.15)
    )
    short_hold = (
        (features["short_score"] >= float(params["trend_exit_threshold_short"]))
        & (features["ema_stack_down"] == 1)
        & (funding >= -float(params["funding_extreme_abs_max_short"]) * 1.35)
        & (activity <= float(params["activity_spike_max_short"]) * 1.15)
    )
    unstable = high_vol | (activity > max(float(params["activity_spike_max_long"]), float(params["activity_spike_max_short"])) * 1.2)
    unstable = unstable | (np.abs(funding) > max(float(params["funding_extreme_abs_max_long"]), float(params["funding_extreme_abs_max_short"])) * 1.5)

    regimes = np.zeros(features["long_score"].shape, dtype=np.int8)
    regimes[unstable] = 3
    regimes[long_hold & ~unstable] = 1
    regimes[short_hold & ~unstable] = -1
    regimes[(long_hold & short_hold) | (long_entry & short_entry)] = 0
    return {
        "regime": regimes,
        "long_entry": long_entry.astype(np.int8),
        "short_entry": short_entry.astype(np.int8),
        "long_hold": long_hold.astype(np.int8),
        "short_hold": short_hold.astype(np.int8),
        "unstable": unstable.astype(np.int8),
    }


def percentile_rank(values: Sequence[float], target: float) -> float:
    if not values:
        return 0.0
    less = sum(1 for value in values if value <= target)
    return less / len(values)


def make_param_space() -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    size_profiles = [
        {"base_size_long": 0.1, "base_size_short": 0.12, "multiplier_long": 1.35, "multiplier_short": 1.45, "max_layers_long": 3, "max_layers_short": 3, "add_step_long": 0.009, "add_step_short": 0.008, "max_position_pct": 0.72},
        {"base_size_long": 0.12, "base_size_short": 0.12, "multiplier_long": 1.45, "multiplier_short": 1.45, "max_layers_long": 3, "max_layers_short": 3, "add_step_long": 0.008, "add_step_short": 0.008, "max_position_pct": 0.8},
        {"base_size_long": 0.12, "base_size_short": 0.14, "multiplier_long": 1.55, "multiplier_short": 1.6, "max_layers_long": 3, "max_layers_short": 4, "add_step_long": 0.008, "add_step_short": 0.007, "max_position_pct": 0.8},
        {"base_size_long": 0.1, "base_size_short": 0.14, "multiplier_long": 1.35, "multiplier_short": 1.55, "max_layers_long": 4, "max_layers_short": 4, "add_step_long": 0.009, "add_step_short": 0.007, "max_position_pct": 0.76},
    ]
    regime_profiles = [
        {"trend_entry_threshold_long": 0.18, "trend_exit_threshold_long": 0.08, "trend_entry_threshold_short": 0.14, "trend_exit_threshold_short": 0.06, "adx_min_long": 14.0, "adx_min_short": 16.0, "funding_extreme_abs_max_long": 0.00008, "funding_extreme_abs_max_short": 0.00008, "oi_change_min_long": -0.001, "oi_change_min_short": 0.0, "activity_min_long": 0.85, "activity_min_short": 0.95, "activity_spike_max_long": 2.6, "activity_spike_max_short": 2.9},
        {"trend_entry_threshold_long": 0.2, "trend_exit_threshold_long": 0.09, "trend_entry_threshold_short": 0.16, "trend_exit_threshold_short": 0.07, "adx_min_long": 15.0, "adx_min_short": 16.0, "funding_extreme_abs_max_long": 0.00007, "funding_extreme_abs_max_short": 0.00009, "oi_change_min_long": 0.0, "oi_change_min_short": 0.001, "activity_min_long": 0.9, "activity_min_short": 0.95, "activity_spike_max_long": 2.8, "activity_spike_max_short": 3.0},
        {"trend_entry_threshold_long": 0.22, "trend_exit_threshold_long": 0.1, "trend_entry_threshold_short": 0.18, "trend_exit_threshold_short": 0.08, "adx_min_long": 16.0, "adx_min_short": 17.0, "funding_extreme_abs_max_long": 0.00006, "funding_extreme_abs_max_short": 0.00008, "oi_change_min_long": 0.001, "oi_change_min_short": 0.001, "activity_min_long": 0.95, "activity_min_short": 1.0, "activity_spike_max_long": 2.7, "activity_spike_max_short": 2.8},
    ]
    exit_profiles = [
        {"partial_take_profit_pct_long": 0.03, "partial_take_profit_pct_short": 0.035, "partial_take_profit_share_long": 0.35, "partial_take_profit_share_short": 0.35, "trailing_activation_pct_long": 0.025, "trailing_activation_pct_short": 0.03, "trailing_stop_gap_long": 0.018, "trailing_stop_gap_short": 0.02, "stop_loss_pct_long": 0.08, "stop_loss_pct_short": 0.08},
        {"partial_take_profit_pct_long": 0.04, "partial_take_profit_pct_short": 0.04, "partial_take_profit_share_long": 0.4, "partial_take_profit_share_short": 0.4, "trailing_activation_pct_long": 0.03, "trailing_activation_pct_short": 0.035, "trailing_stop_gap_long": 0.02, "trailing_stop_gap_short": 0.022, "stop_loss_pct_long": 0.1, "stop_loss_pct_short": 0.1},
        {"partial_take_profit_pct_long": 0.05, "partial_take_profit_pct_short": 0.045, "partial_take_profit_share_long": 0.45, "partial_take_profit_share_short": 0.4, "trailing_activation_pct_long": 0.04, "trailing_activation_pct_short": 0.04, "trailing_stop_gap_long": 0.025, "trailing_stop_gap_short": 0.024, "stop_loss_pct_long": 0.11, "stop_loss_pct_short": 0.1},
    ]
    for size_profile in size_profiles:
        for regime_profile in regime_profiles:
            for exit_profile in exit_profiles:
                for leverage in (2, 3, 4):
                    for add_step_type in ("fixed_pct", "atr_multiple"):
                        candidates.append({**size_profile, **regime_profile, **exit_profile, "add_step_type": add_step_type, "cooldown_bars": 1, "max_loss_per_cycle": 0.2, "exit_on_reversal": True, "leverage": leverage})
    return candidates


def summarize_param_space(params_list: Sequence[Dict[str, object]]) -> Dict[str, object]:
    numeric_keys = [
        "base_size_long",
        "base_size_short",
        "multiplier_long",
        "multiplier_short",
        "max_layers_long",
        "max_layers_short",
        "add_step_long",
        "add_step_short",
        "stop_loss_pct_long",
        "stop_loss_pct_short",
        "partial_take_profit_pct_long",
        "partial_take_profit_pct_short",
        "partial_take_profit_share_long",
        "partial_take_profit_share_short",
        "trailing_activation_pct_long",
        "trailing_activation_pct_short",
        "trailing_stop_gap_long",
        "trailing_stop_gap_short",
        "trend_entry_threshold_long",
        "trend_entry_threshold_short",
        "trend_exit_threshold_long",
        "trend_exit_threshold_short",
        "adx_min_long",
        "adx_min_short",
        "funding_extreme_abs_max_long",
        "funding_extreme_abs_max_short",
        "oi_change_min_long",
        "oi_change_min_short",
        "activity_min_long",
        "activity_min_short",
        "activity_spike_max_long",
        "activity_spike_max_short",
        "cooldown_bars",
        "max_position_pct",
        "max_loss_per_cycle",
        "leverage",
    ]
    categorical_keys = ["add_step_type", "exit_on_reversal"]
    summary: Dict[str, object] = {}
    for key in numeric_keys:
        values = sorted({float(item[key]) for item in params_list})
        summary[key] = [values[0], values[-1]] if len(values) > 1 else values
    for key in categorical_keys:
        summary[key] = sorted({item[key] for item in params_list})
    return summary


@dataclass
class Trade:
    direction: int
    entry_time: int
    exit_time: int
    pnl: float
    holding_hours: float
    regime: int
    layers: int
    max_adverse: float


def backtest(
    hourly: Dict[str, np.ndarray],
    features: Dict[str, np.ndarray],
    params: Dict[str, object],
    start_index: int,
    end_index: int,
    fee_rate: float,
    slippage_rate: float,
) -> Dict[str, object]:
    regimes = classify_regimes(features, params["trend_entry_threshold"], params["trend_exit_threshold"])
    close_price = hourly["close"]
    high_price = hourly["high"]
    low_price = hourly["low"]
    time_array = hourly["time"]
    atr_values = features["atr_1h"]
    funding_hourly = features["funding_hourly"]
    leverage = float(params["leverage"])

    equity = 1.0
    equity_curve = [equity]
    trades: List[Trade] = []
    position_direction = 0
    avg_entry = 0.0
    total_size = 0.0
    current_layers = 0
    next_add_price = 0.0
    cooldown_until = start_index
    entry_index = start_index
    max_adverse = 0.0
    liquidation_events = 0
    regime_counts = {"up": 0, "down": 0, "range": 0, "unstable": 0}

    def compute_add_step(index: int, direction: int) -> float:
        step_type = params["add_step_type"]
        if step_type == "fixed_pct":
            return params["add_step_long"] if direction > 0 else params["add_step_short"]
        if step_type == "atr_multiple":
            atr_pct = atr_values[index] / close_price[index] if close_price[index] else 0.0
            multiple = params["add_step_long"] if direction > 0 else params["add_step_short"]
            return atr_pct * multiple
        atr_pct = atr_values[index] / close_price[index] if close_price[index] else 0.0
        realized_vol = features["std_1h"][index] if not math.isnan(features["std_1h"][index]) else 0.0
        return max(0.008, min(0.025, 0.7 * atr_pct + 2.0 * realized_vol))

    def close_position(index: int, reason: str) -> None:
        nonlocal equity, position_direction, avg_entry, total_size, current_layers, next_add_price, cooldown_until, max_adverse, entry_index, liquidation_events
        if position_direction == 0 or total_size <= 0:
            return
        exit_price = close_price[index] * (1 - slippage_rate * position_direction)
        gross_return = position_direction * (exit_price - avg_entry) / avg_entry
        funding_cost = np.sum(funding_hourly[entry_index : index + 1]) * position_direction * -1
        cost = 2 * fee_rate * total_size + slippage_rate * 2 * total_size
        effective_exposure = total_size * leverage
        pnl = gross_return * effective_exposure - cost + funding_cost * effective_exposure
        equity *= 1 + pnl
        if equity <= 0:
            liquidation_events += 1
            equity = 1e-6
        trade_regime = int(regimes[entry_index]) if entry_index < len(regimes) else 0
        trades.append(
            Trade(
                direction=position_direction,
                entry_time=int(time_array[entry_index]),
                exit_time=int(time_array[index]),
                pnl=float(pnl),
                holding_hours=float(index - entry_index + 1),
                regime=trade_regime,
                layers=current_layers,
                max_adverse=float(max_adverse),
            )
        )
        position_direction = 0
        avg_entry = 0.0
        total_size = 0.0
        current_layers = 0
        next_add_price = 0.0
        max_adverse = 0.0
        cooldown_until = index + int(params["cooldown_bars"])

    for index in range(start_index, end_index):
        regime = int(regimes[index])
        if regime == 1:
            regime_counts["up"] += 1
        elif regime == -1:
            regime_counts["down"] += 1
        elif regime == 0:
            regime_counts["range"] += 1
        else:
            regime_counts["unstable"] += 1

        if position_direction != 0:
            effective_exposure = total_size * leverage
            unrealized = position_direction * (close_price[index] - avg_entry) / avg_entry * effective_exposure
            max_adverse = min(max_adverse, unrealized)
            if unrealized <= -float(params["max_loss_per_cycle"]):
                close_position(index, "max_loss")
            elif unrealized <= -float(params["stop_loss_pct"]):
                close_position(index, "stop")
            elif unrealized >= float(params["take_profit_pct"]):
                close_position(index, "take_profit")
            elif bool(params["exit_on_reversal"]) and ((position_direction > 0 and regime <= 0) or (position_direction < 0 and regime >= 0)):
                close_position(index, "reversal")
            elif position_direction > 0 and current_layers < int(params["max_layers_long"]) and low_price[index] <= next_add_price:
                add_size = float(params["base_size_long"]) * (float(params["multiplier_long"]) ** current_layers)
                projected_size = total_size + add_size
                if projected_size <= float(params["max_position_pct"]):
                    avg_entry = (avg_entry * total_size + next_add_price * add_size) / projected_size
                    total_size = projected_size
                    current_layers += 1
                    next_add_price = avg_entry * (1 - compute_add_step(index, 1))
            elif position_direction < 0 and current_layers < int(params["max_layers_short"]) and high_price[index] >= next_add_price:
                add_size = float(params["base_size_short"]) * (float(params["multiplier_short"]) ** current_layers)
                projected_size = total_size + add_size
                if projected_size <= float(params["max_position_pct"]):
                    avg_entry = (avg_entry * total_size + next_add_price * add_size) / projected_size
                    total_size = projected_size
                    current_layers += 1
                    next_add_price = avg_entry * (1 + compute_add_step(index, -1))

        if position_direction == 0 and index >= cooldown_until:
            if regime == 1:
                total_size = float(params["base_size_long"])
                avg_entry = close_price[index] * (1 + slippage_rate)
                position_direction = 1
                current_layers = 1
                entry_index = index
                next_add_price = avg_entry * (1 - compute_add_step(index, 1))
            elif regime == -1:
                total_size = float(params["base_size_short"])
                avg_entry = close_price[index] * (1 - slippage_rate)
                position_direction = -1
                current_layers = 1
                entry_index = index
                next_add_price = avg_entry * (1 + compute_add_step(index, -1))

        equity_curve.append(equity if position_direction == 0 else equity * (1 + position_direction * total_size * leverage * (close_price[index] - avg_entry) / avg_entry))

        if position_direction != 0:
            margin_buffer = total_size / leverage if leverage else total_size
            liquidation_threshold = max(0.12, margin_buffer * 0.9)
            if abs(max_adverse) >= liquidation_threshold:
                liquidation_events += 1

    if position_direction != 0:
        close_position(end_index - 1, "end")

    equity_array = np.array(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(equity_array)
    drawdowns = equity_array / peaks - 1.0
    pnl_values = np.array([trade.pnl for trade in trades], dtype=np.float64) if trades else np.array([], dtype=np.float64)
    returns = np.diff(np.log(np.clip(equity_array, 1e-9, None)))
    sharpe = 0.0
    sortino = 0.0
    if len(returns) > 5 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * math.sqrt(24 * 365))
        downside = returns[returns < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino = float(np.mean(returns) / np.std(downside) * math.sqrt(24 * 365))

    win_rate = float(np.mean(pnl_values > 0)) if len(pnl_values) else 0.0
    gross_profit = float(np.sum(pnl_values[pnl_values > 0])) if len(pnl_values) else 0.0
    gross_loss = float(-np.sum(pnl_values[pnl_values < 0])) if len(pnl_values) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    avg_hold = float(np.mean([trade.holding_hours for trade in trades])) if trades else 0.0
    worst_trade = float(np.min(pnl_values)) if len(pnl_values) else 0.0
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0
    liquidation_risk = min(1.0, liquidation_events / max(1, len(trades))) if trades else 0.0
    param_instability_penalty = 0.0
    monthly_return = float((equity_array[-1] ** (30.0 / max(1.0, (end_index - start_index) / 24.0))) - 1.0)
    objective = (
        0.9 * (equity_array[-1] - 1.0)
        + 1.2 * monthly_return
        - 1.4 * abs(max_drawdown)
        - 0.18 * np.std(returns)
        - 0.0015 * len(trades)
        - 1.9 * liquidation_risk
        - param_instability_penalty
    )

    regime_trade_stats = {
        "Uptrend": summarize_trades([trade for trade in trades if trade.regime == 1]),
        "Downtrend": summarize_trades([trade for trade in trades if trade.regime == -1]),
        "Range": summarize_trades([trade for trade in trades if trade.regime == 0]),
        "HighVolatilityUnstable": summarize_trades([trade for trade in trades if trade.regime == 3]),
    }
    return {
        "params": params,
        "total_return": float(equity_array[-1] - 1.0),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "trade_count": int(len(trades)),
        "avg_holding_hours": avg_hold,
        "worst_trade": worst_trade,
        "sharpe": sharpe,
        "sortino": sortino,
        "monthly_avg_return": monthly_return,
        "liquidation_risk": float(liquidation_risk),
        "objective": float(objective),
        "regime_trade_stats": regime_trade_stats,
        "regime_bar_counts": regime_counts,
        "trades": trades,
    }


def summarize_trades(trades: Sequence[Trade]) -> Dict[str, float]:
    if not trades:
        return {"count": 0, "return_sum": 0.0, "win_rate": 0.0}
    pnls = np.array([trade.pnl for trade in trades], dtype=np.float64)
    return {
        "count": int(len(trades)),
        "return_sum": float(np.sum(pnls)),
        "win_rate": float(np.mean(pnls > 0)),
    }


def walk_forward_search(hourly: Dict[str, np.ndarray], features: Dict[str, np.ndarray], fee_rate: float, slippage_rate: float) -> Dict[str, object]:
    params_list = make_param_space()
    search_ranges = summarize_param_space(params_list)
    size = len(hourly["time"])
    train_end = int(size * 0.55)
    validation_end = int(size * 0.78)
    candidates: List[Dict[str, object]] = []
    for params in params_list:
        train_result = backtest(hourly, features, params, 120, train_end, fee_rate, slippage_rate)
        validation_result = backtest(hourly, features, params, train_end, validation_end, fee_rate, slippage_rate)
        stability_gap = abs(train_result["objective"] - validation_result["objective"])
        if train_result["liquidation_risk"] > 0.12 or validation_result["liquidation_risk"] > 0.12:
            continue
        if train_result["max_drawdown"] < -0.4 or validation_result["max_drawdown"] < -0.35:
            continue
        if train_result["worst_trade"] < -0.2 or validation_result["worst_trade"] < -0.2:
            continue
        combined_score = (
            0.5 * validation_result["objective"]
            + 0.35 * train_result["objective"]
            + 0.15 * validation_result["monthly_avg_return"]
            - 0.45 * stability_gap
        )
        candidates.append(
            {
                "params": params,
                "train": train_result,
                "validation": validation_result,
                "combined_score": combined_score,
                "stability_gap": stability_gap,
            }
        )
    if not candidates:
        raise RuntimeError("No parameter candidates survived safety filters")
    candidates.sort(key=lambda item: item["combined_score"], reverse=True)
    top_candidates = candidates[:6]
    test_results = []
    for candidate in top_candidates:
        test_result = backtest(hourly, features, candidate["params"], validation_end, size, fee_rate, slippage_rate)
        test_results.append({**candidate, "test": test_result})
    test_results.sort(
        key=lambda item: item["test"]["objective"] + 0.25 * item["test"]["monthly_avg_return"] - 0.55 * item["stability_gap"],
        reverse=True,
    )
    best = test_results[0]
    full_year_result = backtest(hourly, features, best["params"], 120, size, fee_rate, slippage_rate)
    return {
        "searched_candidates": len(params_list),
        "survivors": len(candidates),
        "search_ranges": search_ranges,
        "best": best,
        "full_year": full_year_result,
        "train_end": train_end,
        "validation_end": validation_end,
        "top_candidates": test_results,
    }


def current_regime_summary(hourly: Dict[str, np.ndarray], features: Dict[str, np.ndarray], params: Dict[str, object]) -> Dict[str, object]:
    regimes = classify_regimes(features, params["trend_entry_threshold"], params["trend_exit_threshold"])
    last_index = len(regimes) - 1
    regime_map = {1: "Uptrend", -1: "Downtrend", 0: "Range", 3: "HighVolatilityUnstable", 2: "Range"}
    current = regime_map.get(int(regimes[last_index]), "Range")
    conclusion = {
        "current_regime": current,
        "trend_score": float(features["trend_score"][last_index]),
        "adx_1h": float(features["adx_1h"][last_index]),
        "ema_stack_up": bool(features["ema_stack_up"][last_index]),
        "ema_stack_down": bool(features["ema_stack_down"][last_index]),
        "volatility_flag": bool(features["is_high_vol"][last_index]),
        "time": ms_to_iso(int(hourly["time"][last_index])),
    }
    bars_4h = aggregate_bars(hourly, 4)
    bars_1d = aggregate_bars(hourly, 24)
    ema_4h_21 = ema(bars_4h["close"], 21)
    ema_4h_55 = ema(bars_4h["close"], 55)
    ema_1d_20 = ema(bars_1d["close"], 20)
    ema_1d_50 = ema(bars_1d["close"], 50)
    conclusion["timeframe_view"] = {
        "1h": "bullish" if features["ema_stack_up"][last_index] else "bearish" if features["ema_stack_down"][last_index] else "neutral",
        "4h": "bullish" if ema_4h_21[-1] > ema_4h_55[-1] else "bearish" if ema_4h_21[-1] < ema_4h_55[-1] else "neutral",
        "1d": "bullish" if ema_1d_20[-1] > ema_1d_50[-1] else "bearish" if ema_1d_20[-1] < ema_1d_50[-1] else "neutral",
    }
    return conclusion


def build_final_json(config: ResearchConfig, params: Dict[str, object], risk_flags: List[str]) -> Dict[str, object]:
    return {
        "symbol": config.symbol,
        "market_type": config.market_type,
        "mode": "research_only",
        "timeframes": ["1h", "4h", "1d"],
        "trend_entry_threshold": params["trend_entry_threshold"],
        "trend_exit_threshold": params["trend_exit_threshold"],
        "base_size_long": params["base_size_long"],
        "base_size_short": params["base_size_short"],
        "multiplier_long": params["multiplier_long"],
        "multiplier_short": params["multiplier_short"],
        "max_layers_long": params["max_layers_long"],
        "max_layers_short": params["max_layers_short"],
        "add_step_long": params["add_step_long"],
        "add_step_short": params["add_step_short"],
        "add_step_type": params["add_step_type"],
        "stop_loss_pct": params["stop_loss_pct"],
        "take_profit_pct": params["take_profit_pct"],
        "max_position_pct": params["max_position_pct"],
        "cooldown_bars": params["cooldown_bars"],
        "max_loss_per_cycle": params["max_loss_per_cycle"],
        "exit_on_reversal": params["exit_on_reversal"],
        "leverage": params["leverage"],
        "objective": "expected_return - drawdown_penalty - volatility_penalty - overtrading_penalty - instability_penalty - liquidation_penalty",
        "risk_flags": risk_flags,
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def baseline_return(hourly: Dict[str, np.ndarray], start_index: int, end_index: int, fee_rate: float, slippage_rate: float) -> Dict[str, float]:
    entry_price = hourly["close"][start_index] * (1 + slippage_rate)
    exit_price = hourly["close"][end_index - 1] * (1 - slippage_rate)
    gross = exit_price / entry_price - 1.0
    net = gross - 2 * fee_rate
    return {
        "buy_and_hold_return": float(net),
        "price_change": float(hourly["close"][end_index - 1] / hourly["close"][start_index] - 1.0),
    }


def period_backtest(hourly: Dict[str, np.ndarray], features: Dict[str, np.ndarray], params: Dict[str, object], start_ms: Optional[int], end_ms: Optional[int], fee_rate: float, slippage_rate: float) -> Optional[Dict[str, object]]:
    if start_ms is None or end_ms is None:
        return None
    indices = np.where((hourly["time"] >= start_ms) & (hourly["time"] <= end_ms))[0]
    if len(indices) < 200:
        return None
    return backtest(hourly, features, params, int(indices[0]), int(indices[-1]), fee_rate, slippage_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--slippage-rate", type=float, default=0.0003)
    parser.add_argument("--leverage", type=int, default=2)
    args = parser.parse_args()

    requested_start_ms = iso_to_ms(args.start)
    requested_end_ms = iso_to_ms(args.end)
    end_ms = requested_end_ms or utc_now_ms()
    full_year_start_ms = end_ms - 365 * MS_IN_DAY
    warmup_start_ms = full_year_start_ms
    if requested_start_ms is not None:
        warmup_start_ms = min(warmup_start_ms, requested_start_ms - 90 * MS_IN_DAY)
    config = ResearchConfig(
        symbol=args.symbol,
        market_type=DEFAULT_MARKET_TYPE,
        start_ms=warmup_start_ms,
        end_ms=end_ms,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
    )

    hourly_rows = chunked_kline_fetch(config.symbol, "1h", config.start_ms, config.end_ms)
    if len(hourly_rows) < 1000:
        raise RuntimeError("Insufficient hourly bars fetched from Binance")
    funding_rows = chunked_funding_fetch(config.symbol, config.start_ms, config.end_ms)
    hourly = rows_to_arrays(hourly_rows)
    data_quality = validate_data(hourly, MS_IN_HOUR)
    features = build_features(hourly, funding_rows)
    search = walk_forward_search(hourly, features, config.fee_rate, config.slippage_rate)
    best_params = dict(search["best"]["params"])
    best_params["leverage"] = args.leverage
    full_year_indices = np.where((hourly["time"] >= full_year_start_ms) & (hourly["time"] <= end_ms))[0]
    if len(full_year_indices) < 500:
        raise RuntimeError("Insufficient bars for last-year evaluation window")
    full_year_start_index = int(full_year_indices[0])
    full_year_end_index = int(full_year_indices[-1]) + 1
    full_year_result = backtest(hourly, features, best_params, full_year_start_index, full_year_end_index, config.fee_rate, config.slippage_rate)
    full_year_baseline = baseline_return(hourly, full_year_start_index, full_year_end_index, config.fee_rate, config.slippage_rate)
    current_regime = current_regime_summary(hourly, features, best_params)
    period_result = period_backtest(hourly, features, best_params, requested_start_ms, requested_end_ms, config.fee_rate, config.slippage_rate)
    period_baseline = None
    if requested_start_ms is not None and requested_end_ms is not None:
        indices = np.where((hourly["time"] >= requested_start_ms) & (hourly["time"] <= requested_end_ms))[0]
        if len(indices) >= 200:
            period_baseline = baseline_return(hourly, int(indices[0]), int(indices[-1]) + 1, config.fee_rate, config.slippage_rate)

    risk_flags: List[str] = []
    if full_year_result["max_drawdown"] < -0.2:
        risk_flags.append("max_drawdown_above_20pct")
    if full_year_result["liquidation_risk"] > 0.03:
        risk_flags.append("non_trivial_liquidation_risk")
    if full_year_result["profit_factor"] < 1.15:
        risk_flags.append("weak_profit_factor")
    if full_year_result["trade_count"] < 20:
        risk_flags.append("low_trade_count")
    if search["best"]["stability_gap"] > 0.2:
        risk_flags.append("parameter_instability")
    live_advisable = not risk_flags
    monthly_target = 0.2
    monthly_target_met = full_year_result["monthly_avg_return"] >= monthly_target
    if not monthly_target_met:
        risk_flags.append("monthly_target_not_met")
        live_advisable = False

    final_json = build_final_json(config, best_params, risk_flags)
    final_json["leverage"] = args.leverage

    report = {
        "data_quality": data_quality,
        "evaluation_windows": {
            "fetched_data_start": ms_to_iso(config.start_ms),
            "fetched_data_end": ms_to_iso(config.end_ms),
            "last_year_start": ms_to_iso(full_year_start_ms),
            "last_year_end": ms_to_iso(end_ms),
            "requested_start": None if requested_start_ms is None else ms_to_iso(requested_start_ms),
            "requested_end": None if requested_end_ms is None else ms_to_iso(requested_end_ms),
        },
        "market_regime": current_regime,
        "strategy_branches": {
            "Uptrend": "Long-only layered pullback entries; exit on reversal, stop-loss, or profit target.",
            "Downtrend": "Short-only layered rebound entries; exit on reversal, stop-loss, or profit target.",
            "Range": "Martingale disabled by default.",
            "HighVolatilityUnstable": "Strategy disabled or risk reduced to zero in this research configuration.",
        },
        "optimization": {
            "searched_candidates": search["searched_candidates"],
            "survivors": search["survivors"],
            "search_ranges": search["search_ranges"],
            "best_params": best_params,
            "selection_logic": "validation-first ranking with drawdown, liquidation, and stability filters",
            "stability_gap": search["best"]["stability_gap"],
        },
        "last_year_backtest": {
            **{key: value for key, value in full_year_result.items() if key not in {"params", "trades"}},
            "baseline": full_year_baseline,
        },
        "requested_period_backtest": None if period_result is None else {
            **{key: value for key, value in period_result.items() if key not in {"params", "trades"}},
            "baseline": period_baseline,
        },
        "risk": {
            "flags": risk_flags,
            "live_advisable": live_advisable,
            "verdict": "Not recommended for live deployment" if not live_advisable else "Research candidate only; still requires further paper trading",
        },
        "target_assessment": {
            "target_monthly_avg_return": monthly_target,
            "achieved_monthly_avg_return": full_year_result["monthly_avg_return"],
            "target_met": monthly_target_met,
        },
        "json_config": final_json,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())