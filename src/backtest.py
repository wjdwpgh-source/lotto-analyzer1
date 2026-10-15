"""6단계: 전략 검증 (백테스트).

각 회차마다 '그 회차 이전 데이터만' 보고 번호를 만든 뒤,
실제 당첨번호와 비교합니다. 미래 데이터를 쓰지 않으므로 공정한 비교입니다.

우연 범위 계산 (v2)
-------------------
같은 회차의 여러 게임은 같은 당첨번호와 비교되므로 서로 독립이 아닙니다.
특히 '많이 나온 번호' 같은 전략은 한 회차의 게임들이 같은 번호를 공유해서,
게임 수만큼 독립 시행으로 보면 우연 범위가 실제보다 좁게 나옵니다.
v2에서는 '추첨이 완전 무작위'라는 가정 아래, 각 회차에 실제로 만든 게임 묶음의
일치 개수 분산을 정확히 계산해 전략마다 따로 우연 범위를 구합니다.
"""
import numpy as np
import pandas as pd

from . import analyzer, generator, probability

NUM_COLS = analyzer.NUM_COLS

# 무작위 추첨에서 번호 x가 뽑힐 확률 p, 서로 다른 두 번호가 함께 뽑힐 확률 q
P1 = 6 / 45
P2 = (6 * 5) / (45 * 44)
MEAN = 6 * P1  # 게임 1장의 평균 일치 개수 = 0.8
Z = 1.96  # 약 95% 범위


def rank_of(ticket, winning, bonus):
    m = len(set(ticket) & set(winning))
    if m == 6:
        return m, 1
    if m == 5:
        return m, 2 if bonus in ticket else 3
    if m == 4:
        return m, 4
    if m == 3:
        return m, 5
    return m, None


def round_sum_variance(tickets) -> float:
    """한 회차에 산 게임 묶음의 '일치 개수 합'이 무작위 추첨에서 갖는 분산 (정확값).

    합 = Σ_x c_x · I(x가 뽑힘),  c_x = 번호 x가 들어간 게임 수
    Var = p(1-p)·Σc_x² + (q - p²)·[(Σc_x)² - Σc_x²]
    게임끼리 번호가 겹칠수록 분산이 커져 우연 범위가 넓어집니다.
    """
    c = np.zeros(46)
    for t in tickets:
        c[list(t)] += 1
    s1, s2 = c.sum(), (c ** 2).sum()
    return P1 * (1 - P1) * s2 + (P2 - P1 ** 2) * (s1 ** 2 - s2)


def run(df: pd.DataFrame, test_rounds=100, tickets_per_round=5,
        strategies=None, seed=42, progress=None) -> pd.DataFrame:
    strategies = strategies or list(generator.STRATEGIES)
    df = df.sort_values("round").reset_index(drop=True)
    feats = analyzer.features(df)
    start = max(len(df) - test_rounds, 100)  # 최소 100회차 이상 학습 데이터 확보
    stats = {s: {"tickets": 0, "matches": 0, "var_sum": 0.0,
                 "ranks": {r: 0 for r in range(1, 6)}, "won": 0}
             for s in strategies}

    for step, i in enumerate(range(start, len(df))):
        hist, target = df.iloc[:i], df.iloc[i]
        winning = target[NUM_COLS].astype(int).tolist()
        bonus = int(target["bonus"])
        lo, hi = np.percentile(feats["sum"].iloc[:i], [10, 90])
        rules = {"odd": (2, 4), "high": (2, 4), "sum": (int(lo), int(hi)), "ac_min": 7}
        for s_idx, s in enumerate(strategies):
            rng = np.random.default_rng(seed + i * 31 + s_idx)
            w = generator._weights(hist, s)
            tickets = generator.generate(hist, s, tickets_per_round, rng=rng,
                                         rules=rules, weights=w)
            st = stats[s]
            st["var_sum"] += round_sum_variance(tickets)
            for t in tickets:
                m, r = rank_of(t, winning, bonus)
                st["tickets"] += 1
                st["matches"] += m
                if r:
                    st["ranks"][r] += 1
                    st["won"] += int(target[f"win{r}_amount"])
        if progress:
            progress((step + 1) / (len(df) - start))

    rows = []
    for s, st in stats.items():
        n = st["tickets"]
        spent = n * probability.TICKET_PRICE
        avg = st["matches"] / n
        se = st["var_sum"] ** 0.5 / n  # 평균 일치 개수의 표준오차 (무작위 추첨 가정)
        lo, hi = MEAN - Z * se, MEAN + Z * se
        hits = sum(st["ranks"].values())
        rows.append({
            "전략": generator.STRATEGIES[s],
            "구매 장수": n,
            "평균 일치 개수": round(avg, 3),
            "우연 범위 하한": round(lo, 3),
            "우연 범위 상한": round(hi, 3),
            "z값": round((avg - MEAN) / se, 2),
            "판정": "우연 범위 안" if lo <= avg <= hi else "범위 밖",
            "5등 이상 적중률": hits / n,
            **{f"{r}등": st["ranks"][r] for r in range(1, 6)},
            "회수율": st["won"] / spent if spent else 0,
        })
    return pd.DataFrame(rows)


def theory(n_tickets: int = 1) -> dict:
    """무작위로 샀을 때의 이론값과, 서로 독립인 n_tickets장 기준 우연 범위(약 95%).
    전략별 정확한 범위는 run() 결과의 '우연 범위 하한/상한'을 쓰세요."""
    p_any = sum(probability.RANK_CASES.values()) / probability.TOTAL
    var = 6 * P1 * (39 / 45) * (39 / 44)  # 초기하분포 분산
    se = (var / n_tickets) ** 0.5
    return {"평균 일치 개수": MEAN, "5등 이상 적중률": p_any,
            "우연 범위": (MEAN - Z * se, MEAN + Z * se)}
