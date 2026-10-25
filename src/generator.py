"""5단계: 전략별 번호 생성기.

어떤 전략도 당첨 확률을 높이지 못합니다(모든 조합 1/8,145,060).
전략은 '어떤 기준으로 고를지'의 취향이며, 효과는 backtest.py로 검증합니다.
"""
import numpy as np
import pandas as pd

from . import analyzer

STRATEGIES = {
    "random": "완전 무작위",
    "hot": "많이 나온 번호 위주",
    "cold": "오래 안 나온 번호 위주",
    "pattern": "역대 흔한 패턴만 (홀짝·고저·합계·AC)",
    "mixed": "많이 나온 번호 + 흔한 패턴",
    "recent5": "최근 5회 흐름 분석 조합",
}

RECENT_N = 5            # 최근 몇 회를 볼지
RECENT_OVERLAP = (2, 4)  # 최근 5회에 나온 번호를 몇 개 넣을지 (역대 약 80%가 이 범위)
LAST_REPEAT_MAX = 2      # 바로 전 회차 번호는 최대 몇 개까지 (역대 약 98%가 이 범위)


def recent_info(df, n: int = RECENT_N) -> dict:
    """최근 n회 분석: 번호별 출현 횟수, 나온 번호 묶음, 바로 전 회차 번호."""
    last = df.tail(n)
    nums = last[analyzer.NUM_COLS].to_numpy(dtype=int).ravel()
    counts = pd.Series(np.bincount(nums, minlength=46)[1:], index=range(1, 46))
    latest = set(df.iloc[-1][analyzer.NUM_COLS].astype(int))
    return {"counts": counts, "pool": set(counts[counts > 0].index), "latest": latest,
            "rounds": last["round"].astype(int).tolist()}


def overlap_history(df, n: int = RECENT_N) -> pd.Series:
    """역대 각 회차가 그 직전 n회에 나온 번호를 몇 개 포함했는지의 분포(비율)."""
    a = df[analyzer.NUM_COLS].to_numpy(dtype=int)
    ov = [len(set(a[i - n:i].ravel()) & set(a[i])) for i in range(n, len(a))]
    return pd.Series(ov).value_counts(normalize=True).reindex(range(7), fill_value=0)


def passes_recent(nums, info) -> bool:
    k = len(set(nums) & info["pool"])
    return (RECENT_OVERLAP[0] <= k <= RECENT_OVERLAP[1]
            and len(set(nums) & info["latest"]) <= LAST_REPEAT_MAX)


def _weights(df, strategy: str, recent: int = 100) -> np.ndarray:
    if strategy == "recent5":
        # 최근 5회에 여러 번 나온 번호일수록 조금 더 자주 뽑히도록 (안 나온 번호도 기본 1)
        c = recent_info(df)["counts"].to_numpy(dtype=float) + 1
        return c / c.sum()
    if strategy in ("hot", "mixed"):
        f = analyzer.frequency(df, recent).to_numpy(dtype=float) + 1
        return f / f.sum()
    if strategy == "cold":
        g = analyzer.gaps(df).to_numpy(dtype=float) + 1
        return g / g.sum()
    return np.full(45, 1 / 45)


def pattern_rules(df) -> dict:
    """역대 회차에서 흔한 범위 (합계는 중간 80% 구간)."""
    feat = analyzer.features(df)
    lo, hi = np.percentile(feat["sum"], [10, 90])
    return {"odd": (2, 4), "high": (2, 4), "sum": (int(lo), int(hi)), "ac_min": 7}


def passes(nums, rules) -> bool:
    return (rules["odd"][0] <= analyzer.odd_count(nums) <= rules["odd"][1]
            and rules["high"][0] <= analyzer.high_count(nums) <= rules["high"][1]
            and rules["sum"][0] <= sum(nums) <= rules["sum"][1]
            and analyzer.ac_value(nums) >= rules["ac_min"])


def generate(df: pd.DataFrame, strategy="random", count=5, fixed=(), exclude=(),
             rng=None, rules=None, weights=None) -> list:
    """번호 조합 count개를 만듭니다. fixed는 반드시 포함, exclude는 제외."""
    rng = rng or np.random.default_rng()
    fixed = sorted(set(int(n) for n in fixed))
    exclude = set(int(n) for n in exclude) - set(fixed)
    if len(fixed) > 6:
        raise ValueError("고정수는 6개까지만 지정할 수 있습니다.")
    pool = np.array([n for n in range(1, 46) if n not in exclude and n not in fixed])
    need = 6 - len(fixed)
    if len(pool) < need:
        raise ValueError("제외수가 너무 많아 조합을 만들 수 없습니다.")

    w = (weights if weights is not None else _weights(df, strategy))[pool - 1]
    w = w / w.sum()
    use_pattern = strategy in ("pattern", "mixed")
    info = recent_info(df) if strategy == "recent5" else None
    if use_pattern and rules is None:
        rules = pattern_rules(df)

    results, seen = [], set()
    for _ in range(count * 5000):
        pick = tuple(sorted(fixed + rng.choice(pool, need, replace=False, p=w).tolist()))
        if pick in seen:
            continue
        if use_pattern and not passes(pick, rules):
            continue
        if info is not None and not passes_recent(pick, info):
            continue
        seen.add(pick)
        results.append(list(pick))
        if len(results) == count:
            break
    return results


def past_first_prize(df, nums):
    """같은 조합이 과거 1등 번호였으면 그 회차 번호 목록을 돌려줍니다."""
    s = sorted(nums)
    mask = (df[analyzer.NUM_COLS].to_numpy() == np.array(s)).all(axis=1)
    return df.loc[mask, "round"].tolist()
