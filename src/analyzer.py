"""3단계: 역대 당첨번호 통계 분석."""
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
NUMBERS = range(1, 46)
SECTIONS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]


def _recent(df: pd.DataFrame, last_n=None) -> pd.DataFrame:
    return df.tail(last_n) if last_n else df


def frequency(df, last_n=None, include_bonus=False) -> pd.Series:
    """번호별 출현 횟수 (1~45)."""
    cols = NUM_COLS + (["bonus"] if include_bonus else [])
    vals = _recent(df, last_n)[cols].to_numpy().ravel()
    return pd.Series(Counter(vals), dtype=int).reindex(NUMBERS, fill_value=0)


def gaps(df) -> pd.Series:
    """각 번호가 마지막으로 나온 뒤 지난 회차 수 (0 = 직전 회차에 나옴)."""
    last_round = int(df["round"].max())
    out = {}
    for n in NUMBERS:
        hit = df.loc[(df[NUM_COLS] == n).any(axis=1), "round"]
        out[n] = last_round - int(hit.max()) if len(hit) else len(df)
    return pd.Series(out)


def hot_cold(df, last_n=20, k=6):
    """최근 last_n회 기준 많이 나온 번호 / 적게 나온 번호."""
    f = frequency(df, last_n).sort_values(kind="stable")
    return f.index[::-1][:k].tolist(), f.index[:k].tolist()


# ---------- 조합 하나의 특징 ----------
def odd_count(nums) -> int:
    return sum(n % 2 for n in nums)


def high_count(nums) -> int:
    """23~45를 '고'번호로 봅니다."""
    return sum(n >= 23 for n in nums)


def consecutive_pairs(nums) -> int:
    s = sorted(nums)
    return sum(b - a == 1 for a, b in zip(s, s[1:]))


def ac_value(nums) -> int:
    """AC값: 서로 다른 '번호 간 차이'의 개수 - 5. 클수록 번호가 고르게 흩어짐 (0~10)."""
    diffs = {abs(a - b) for a, b in combinations(nums, 2)}
    return len(diffs) - (len(nums) - 1)


def describe(nums) -> dict:
    nums = sorted(nums)
    return {"홀:짝": f"{odd_count(nums)}:{6 - odd_count(nums)}",
            "고:저": f"{high_count(nums)}:{6 - high_count(nums)}",
            "합계": sum(nums), "연번 쌍": consecutive_pairs(nums), "AC값": ac_value(nums)}


# ---------- 역대 회차의 패턴 분포 ----------
def features(df) -> pd.DataFrame:
    nums = df[NUM_COLS].to_numpy()
    return pd.DataFrame({
        "round": df["round"].to_numpy(),
        "odd": [odd_count(r) for r in nums],
        "high": [high_count(r) for r in nums],
        "sum": nums.sum(axis=1),
        "consec": [consecutive_pairs(r) for r in nums],
        "ac": [ac_value(r) for r in nums],
    })


def section_counts(df, last_n=None) -> pd.Series:
    f = frequency(df, last_n)
    return pd.Series({f"{a}~{b}": int(f.loc[a:b].sum()) for a, b in SECTIONS})


def last_digit_counts(df, last_n=None) -> pd.Series:
    f = frequency(df, last_n)
    return f.groupby(f.index % 10).sum().rename(lambda d: f"끝수 {d}")


def top_pairs(df, k=10, last_n=None) -> pd.DataFrame:
    c = Counter()
    for row in _recent(df, last_n)[NUM_COLS].to_numpy():
        c.update(combinations(sorted(row), 2))
    return pd.DataFrame([{"번호 쌍": f"{a} - {b}", "함께 나온 횟수": n}
                         for (a, b), n in c.most_common(k)])


def chi_square(df, last_n=None):
    """출현 빈도가 '모든 번호 균등'과 통계적으로 다른지 검정.
    p값이 0.05보다 크면 '무작위와 구별되는 치우침이 없다'는 뜻입니다."""
    f = frequency(df, last_n)
    expected = np.full(45, f.sum() / 45)
    stat, p = stats.chisquare(f.to_numpy(), expected)
    return float(stat), float(p)
