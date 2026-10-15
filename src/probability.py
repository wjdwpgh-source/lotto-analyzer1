"""4단계: 로또 6/45 확률 계산 (모두 정확한 조합 계산)."""
from math import comb

import pandas as pd

TOTAL = comb(45, 6)  # 8,145,060
TICKET_PRICE = 1000

# 등수별 경우의 수: 내 번호 6개 중 몇 개가 당첨번호와 맞는가
RANK_CASES = {
    1: comb(6, 6),                    # 6개 일치
    2: comb(6, 5) * 1,                # 5개 + 보너스
    3: comb(6, 5) * 38,               # 5개 + 나머지 1개가 보너스가 아닌 38개 중 하나
    4: comb(6, 4) * comb(39, 2),      # 4개 일치
    5: comb(6, 3) * comb(39, 3),      # 3개 일치
}
RANK_CONDITION = {1: "6개 일치", 2: "5개 + 보너스", 3: "5개 일치",
                  4: "4개 일치", 5: "3개 일치"}


def prize_table() -> pd.DataFrame:
    rows = []
    for r, cases in RANK_CASES.items():
        p = cases / TOTAL
        rows.append({"등수": f"{r}등", "조건": RANK_CONDITION[r], "경우의 수": cases,
                     "확률": p, "1/n": round(1 / p, 1)})
    total_cases = sum(RANK_CASES.values())
    rows.append({"등수": "아무 등수", "조건": "5등 이상", "경우의 수": total_cases,
                 "확률": total_cases / TOTAL, "1/n": round(TOTAL / total_cases, 1)})
    return pd.DataFrame(rows)


def prob_with_tickets(rank: int, tickets: int) -> float:
    """서로 다른 조합 tickets장을 샀을 때 해당 등수에 한 번 이상 당첨될 확률(근사)."""
    p = RANK_CASES[rank] / TOTAL
    return 1 - (1 - p) ** tickets


def weeks_to_win(rank: int, tickets_per_week: int, confidence=0.5) -> float:
    """매주 tickets_per_week장씩 살 때, confidence 확률로 한 번 당첨되려면 걸리는 주 수."""
    from math import log
    p_week = prob_with_tickets(rank, tickets_per_week)
    return log(1 - confidence) / log(1 - p_week)


def odd_count_theory() -> pd.Series:
    """6개 중 홀수가 k개일 이론 확률 (1~45 중 홀수 23개, 짝수 22개)."""
    return pd.Series({k: comb(23, k) * comb(22, 6 - k) / TOTAL for k in range(7)})


def high_count_theory() -> pd.Series:
    """6개 중 고번호(23~45, 23개)가 k개일 이론 확률."""
    return odd_count_theory()  # 23개 vs 22개로 구조가 같음


def expected_return(df, last_n=52) -> dict:
    """최근 last_n회 실제 평균 당첨금으로 1장(1,000원)의 기대 회수액 계산 (세전)."""
    recent = df.tail(last_n)
    ev = 0.0
    detail = {}
    for r, cases in RANK_CASES.items():
        avg_amount = recent[f"win{r}_amount"].mean()
        part = avg_amount * cases / TOTAL
        detail[f"{r}등"] = part
        ev += part
    return {"기대 회수액": ev, "회수율": ev / TICKET_PRICE, "등수별": detail}
