"""로또분석기 자가 점검.  실행: run_test.bat  (또는 python selftest.py)

각 기능이 제대로 동작하는지 차례로 확인하고 결과를 [통과]/[실패]로 보여 줍니다.
인터넷이 없으면 '최신 데이터 받기' 항목만 [건너뜀]으로 표시됩니다.
"""
import sys
import time
import traceback
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import analyzer, backtest, collector, database, generator, probability  # noqa: E402

results = []


def check(name):
    def deco(fn):
        t = time.time()
        try:
            msg = fn() or ""
            results.append(("통과", name, msg))
        except SkipTest as e:
            results.append(("건너뜀", name, str(e)))
        except Exception as e:  # noqa: BLE001
            results.append(("실패", name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        status, _, msg = results[-1]
        print(f"[{status}] {name} ({time.time() - t:.1f}초) {msg}", flush=True)
        return fn
    return deco


class SkipTest(Exception):
    pass


df = database.load()


@check("1. 저장된 데이터 불러오기·검증")
def _():
    assert len(df) > 1000, f"회차 수가 너무 적습니다: {len(df)}"
    problems = database.validate(df)
    assert not problems, " / ".join(problems)
    return f"{len(df)}개 회차 (최신 {int(df['round'].max())}회, {df['date'].iloc[-1]})"


@check("2. 최신 데이터 받기 (동행복권)")
def _():
    try:
        new = collector.fetch_all()
    except Exception as e:  # noqa: BLE001
        raise SkipTest(f"접속 실패 — 저장된 데이터로 계속 동작합니다 ({e})")
    assert not database.validate(new), "받은 데이터 검증 실패"
    assert int(new["round"].max()) >= int(df["round"].max())
    same = new[new["round"] <= df["round"].max()].reset_index(drop=True)
    assert (same[analyzer.NUM_COLS].values == df[analyzer.NUM_COLS].values).all(), \
        "저장된 번호와 받은 번호가 다릅니다"
    extra = int(new["round"].max()) - int(df["round"].max())
    return f"{len(new)}개 회차 수신, 저장본과 일치" + (f" (새 회차 {extra}개 있음)" if extra else "")


@check("3. 확률 계산")
def _():
    assert probability.TOTAL == 8_145_060
    expect = {1: 1, 2: 6, 3: 228, 4: 11_115, 5: 182_780}
    assert probability.RANK_CASES == expect, probability.RANK_CASES
    assert abs(probability.odd_count_theory().sum() - 1) < 1e-12
    return "등수별 경우의 수 1 / 6 / 228 / 11,115 / 182,780"


@check("4. 통계 분석")
def _():
    f = analyzer.frequency(df)
    assert f.sum() == len(df) * 6 and len(f) == 45
    assert analyzer.ac_value([1, 2, 3, 4, 5, 6]) == 0
    assert analyzer.ac_value([1, 2, 4, 8, 16, 32]) == 10
    stat, p = analyzer.chi_square(df)
    return f"카이제곱 p값 {p:.3f}"


@check("5. 번호 생성 (전략 5종 + 고정수·제외수)")
def _():
    rng = np.random.default_rng(0)
    for s in generator.STRATEGIES:
        tickets = generator.generate(df, s, 5, fixed=[7], exclude=[1, 2, 3], rng=rng)
        assert len(tickets) == 5, f"{s}: {len(tickets)}게임만 생성"
        for t in tickets:
            assert len(set(t)) == 6 and 7 in t and not {1, 2, 3} & set(t)
            if s in ("pattern", "mixed"):
                assert generator.passes(t, generator.pattern_rules(df))
    return "모든 전략 정상"


@check("6. 우연 범위 공식 (모의 추첨 20만 번과 비교)")
def _():
    rng = np.random.default_rng(1)
    # 번호가 많이 겹치는 묶음(많이 나온 번호 전략과 비슷한 상황)
    tickets = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 8, 9, 10],
               [1, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]
    formula = backtest.round_sum_variance(tickets)
    mask = np.zeros((len(tickets), 46), dtype=bool)
    for i, t in enumerate(tickets):
        mask[i, t] = True
    draws = np.argsort(rng.random((200_000, 45)), axis=1)[:, :6] + 1
    hit = np.zeros((len(draws), 46), dtype=bool)
    np.put_along_axis(hit, draws, True, axis=1)
    sums = (hit[:, None, :] & mask[None]).sum(axis=(1, 2))
    sim = sums.var()
    assert abs(sim - formula) / formula < 0.03, f"공식 {formula:.4f} vs 모의 {sim:.4f}"
    single = backtest.round_sum_variance([[1, 2, 3, 4, 5, 6]])
    hyper = 6 * (6 / 45) * (39 / 45) * (39 / 44)
    assert abs(single - hyper) < 1e-12
    return f"분산 공식 {formula:.4f} ≈ 모의 {sim:.4f}"


@check("7. 전략 검증 (최근 50회 × 5게임)")
def _():
    res = backtest.run(df, 50, 5)
    assert len(res) == len(generator.STRATEGIES)
    assert (res["우연 범위 하한"] < backtest.MEAN).all() and (res["우연 범위 상한"] > backtest.MEAN).all()
    widths = dict(zip(res["전략"], (res["우연 범위 상한"] - res["우연 범위 하한"]).round(3)))
    return "범위 폭: " + ", ".join(f"{k} {v}" for k, v in widths.items())


@check("8. 화면 (5개 탭 실행)")
def _():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        raise SkipTest("streamlit 테스트 도구 없음")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception[0].value
    at.button[1].click().run()  # 번호 만들기
    assert not at.exception and len(at.session_state["tickets"]) == 5
    start = next(b for b in at.button if b.label == "검증 시작")
    at.slider[1].set_value(50)
    start.click().run()
    assert not at.exception, at.exception[0].value
    assert "우연 범위 하한" in at.session_state["bt"][0]
    return "번호 생성·전략 검증 버튼까지 정상"


print()
fails = [r for r in results if r[0] == "실패"]
print("=" * 50)
print(f"통과 {sum(r[0] == '통과' for r in results)} · 실패 {len(fails)} · "
      f"건너뜀 {sum(r[0] == '건너뜀' for r in results)}")
print("모든 점검을 통과했습니다." if not fails else "실패한 항목을 확인해 주세요.")
sys.exit(1 if fails else 0)
