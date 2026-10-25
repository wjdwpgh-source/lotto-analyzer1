"""로또분석기 — PC·스마트폰·태블릿 브라우저에서 동작하는 웹 앱.

실행:  streamlit run app.py
같은 와이파이의 폰/패드에서 접속:  run_mobile.bat 실행 후 화면에 나온 주소로 접속
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analyzer, backtest, database, generator, probability

st.set_page_config(page_title="로또분석기", page_icon="🎱", layout="centered",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- 스타일
BALL_COLORS = [(10, "#FBC400"), (20, "#69C8F2"), (30, "#FF7272"),
               (40, "#AAAAAA"), (45, "#B0D840")]


def ball_color(n: int) -> str:
    return next(c for limit, c in BALL_COLORS if n <= limit)


st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo",
               "Malgun Gothic", sans-serif !important;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 760px; }
h1 { font-weight: 800; letter-spacing: -0.02em; }
.balls { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 2px; }
.ball { width: 40px; height: 40px; border-radius: 50%; display: inline-flex;
  align-items: center; justify-content: center; font-weight: 800; font-size: 16px;
  color: #000; text-shadow: 0 1px 1px rgba(255,255,255,.45);
  box-shadow: inset -3px -4px 6px rgba(0,0,0,.18), inset 3px 3px 6px rgba(255,255,255,.35); }
.plus { font-weight: 700; opacity: .55; margin: 0 2px; }
.meta { font-size: 13px; opacity: .7; margin-bottom: 10px; }
/* 최근 5회 당첨번호 목록 */
.recent { margin: 6px 0 14px; border-top: 1px solid rgba(128,128,128,.25); }
.recent .row { display: flex; align-items: center; gap: 10px; padding: 7px 0;
  border-bottom: 1px solid rgba(128,128,128,.18); }
.recent .rd { min-width: 78px; font-size: 13px; line-height: 1.25; }
.recent .rd b { font-size: 14px; }
.recent .rd span { opacity: .6; font-size: 12px; }
.recent .balls { margin: 0; gap: 4px; flex-wrap: nowrap; }
.recent .ball { width: 30px; height: 30px; font-size: 13px; }
.recent .plus { margin: 0 1px; }
.ticket { padding: 10px 0 12px; border-bottom: 1px dashed rgba(128,128,128,.35); }
.ticket .label { font-size: 12px; opacity: .65; margin-top: 4px; }
/* 탭을 손가락으로 가로 스크롤 */
.stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; scrollbar-width: none; }
.stTabs [data-baseweb="tab"] { white-space: nowrap; padding: 8px 10px; }
/* 버튼을 터치하기 좋은 크기로 */
.stButton button { min-height: 46px; font-weight: 700; }
@media (max-width: 640px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: 1.2rem; }
  h1 { font-size: 1.6rem !important; }
  .ball { width: 34px; height: 34px; font-size: 14px; }
  .balls { gap: 5px; }
  .recent .rd { min-width: 62px; }
  .recent .ball { width: 26px; height: 26px; font-size: 11.5px; }
  .recent .balls { gap: 3px; }
}
/* 번호판: 1~45 공을 9열 격자로 나열해 직접 누르기 */
.st-key-ball_grid [role="toolbar"] { display: grid !important; grid-template-columns: repeat(9, minmax(0, 1fr));
  gap: 6px; max-width: 460px; }
.st-key-ball_grid button[data-variant="pills"] { aspect-ratio: 1; width: 100%; min-height: 0; padding: 0;
  border-radius: 50%; justify-content: center; font-weight: 700; }
.st-key-ball_grid button[data-variant="pills"] p { font-size: 14px; font-weight: 700; }
.st-key-ball_grid button[aria-pressed="true"] { border-color: transparent; }
.st-key-ball_grid button[aria-pressed="true"] p { color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,.35); }
.st-key-ball_grid button[aria-pressed="true"]:nth-child(-n+10) { background: #FBC400; }
.st-key-ball_grid button[aria-pressed="true"]:nth-child(n+11):nth-child(-n+20) { background: #69C8F2; }
.st-key-ball_grid button[aria-pressed="true"]:nth-child(n+21):nth-child(-n+30) { background: #FF7272; }
.st-key-ball_grid button[aria-pressed="true"]:nth-child(n+31):nth-child(-n+40) { background: #AAAAAA; }
.st-key-ball_grid button[aria-pressed="true"]:nth-child(n+41) { background: #B0D840; }
@media (max-width: 640px) {
  .st-key-ball_grid [role="toolbar"] { gap: 4px; }
  .st-key-ball_grid button[data-variant="pills"] p { font-size: 12px; }
}
</style>
""", unsafe_allow_html=True)


def balls_html(nums, bonus=None) -> str:
    html = "".join(f'<span class="ball" style="background:{ball_color(n)}">{n}</span>'
                   for n in nums)
    if bonus is not None:
        html += f'<span class="plus">+</span><span class="ball" style="background:{ball_color(bonus)}">{bonus}</span>'
    return f'<div class="balls">{html}</div>'


PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}


def small_layout(fig, height=300, title=None):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36 if title else 10, b=8),
                      title=title, showlegend=False, dragmode=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def won(n):
    return f"{n:,.0f}원"


# ---------------------------------------------------------------- 데이터
@st.cache_data(ttl=3600, show_spinner=False)
def get_data() -> pd.DataFrame:
    df = database.load()
    if df.empty:
        df, _ = database.update()
    return df


df = get_data()
if df.empty:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.")
    st.stop()

NUM = analyzer.NUM_COLS
latest = df.iloc[-1]

# ---------------------------------------------------------------- 헤더
st.title("🎱 로또분석기")
st.markdown(f'<div class="meta">{int(latest["round"])}회 · {latest["date"]} 추첨</div>',
            unsafe_allow_html=True)
st.markdown(balls_html(latest[NUM].astype(int).tolist(), int(latest["bonus"])),
            unsafe_allow_html=True)
st.caption(f'1등 {int(latest["win1_count"])}명 · 1인당 {won(latest["win1_amount"])}')

RECENT_N = 5  # 최근 몇 회까지 보여줄지


def recent_html(rows) -> str:
    out = []
    for _, r in rows.iterrows():
        date = str(r["date"])[2:].replace("-", ".")  # 2026-10-24 → 26.10.24
        out.append(f'<div class="row"><div class="rd"><b>{int(r["round"])}회</b><br>'
                   f'<span>{date}</span></div>'
                   f'{balls_html(r[NUM].astype(int).tolist(), int(r["bonus"]))}</div>')
    return f'<div class="recent">{"".join(out)}</div>'


st.markdown(f"**최근 {RECENT_N}회 당첨번호**")
st.markdown(recent_html(df.iloc[::-1].head(RECENT_N)), unsafe_allow_html=True)

if st.button("🔄 최신 당첨번호 받기", width="stretch"):
    with st.spinner("동행복권에서 받는 중…"):
        _, msg = database.update()
    get_data.clear()
    st.toast(msg)
    st.rerun()

tab_gen, tab_stat, tab_prob, tab_bt, tab_check = st.tabs(
    ["🎲 번호 생성", "📊 통계", "🎯 확률", "🧪 전략 검증", "🔎 내 번호 조회"])

# ---------------------------------------------------------------- 번호 생성
with tab_gen:
    strategy = st.selectbox("생성 방식", list(generator.STRATEGIES),
                            format_func=lambda k: generator.STRATEGIES[k])
    count = st.slider("몇 게임", 1, 10, 5)
    with st.expander("고정수 · 제외수 지정"):
        fixed = st.multiselect("꼭 넣을 번호", range(1, 46), max_selections=5)
        exclude = st.multiselect("뺄 번호", [n for n in range(1, 46) if n not in fixed])
    if st.button("번호 만들기", type="primary", width="stretch"):
        try:
            st.session_state["tickets"] = generator.generate(df, strategy, count, fixed, exclude)
        except ValueError as e:
            st.error(str(e))
    for i, t in enumerate(st.session_state.get("tickets", []), 1):
        d = analyzer.describe(t)
        label = " · ".join(f"{k} {v}" for k, v in d.items())
        st.markdown(f'<div class="ticket">{balls_html(t)}<div class="label">{chr(64 + i)}게임 · {label}</div></div>',
                    unsafe_allow_html=True)
        if (hit := generator.past_first_prize(df, t)):
            st.warning(f"이 조합은 {hit}회 1등 번호와 같습니다. 다시 나올 확률도 다른 조합과 똑같습니다.")
    st.caption("어떤 방식이든 1등 확률은 조합당 1/8,145,060으로 같습니다. "
               "방식별 실제 성적은 ‘전략 검증’ 탭에서 확인할 수 있습니다.")

# ---------------------------------------------------------------- 통계
with tab_stat:
    period_label = st.radio("기간", ["전체", "최근 100회", "최근 50회", "최근 20회"],
                            horizontal=True)
    last_n = {"전체": None, "최근 100회": 100, "최근 50회": 50, "최근 20회": 20}[period_label]

    freq = analyzer.frequency(df, last_n)
    fig = go.Figure(go.Bar(x=freq.index, y=freq.values,
                           marker_color=[ball_color(n) for n in freq.index],
                           hovertemplate="%{x}번: %{y}회<extra></extra>"))
    expected = freq.sum() / 45
    fig.add_hline(y=expected, line_dash="dot", annotation_text="평균", opacity=.6)
    fig.update_xaxes(dtick=5, fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(small_layout(fig, 280, "번호별 출현 횟수"), width="stretch",
                    config=PLOT_CONFIG)

    hot, cold = analyzer.hot_cold(df, last_n or len(df))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**많이 나온 번호**")
        st.markdown(balls_html(sorted(hot)), unsafe_allow_html=True)
    with c2:
        st.markdown("**적게 나온 번호**")
        st.markdown(balls_html(sorted(cold)), unsafe_allow_html=True)

    st.markdown("**오래 안 나온 번호**")
    gaps = analyzer.gaps(df).sort_values(ascending=False).head(8)
    st.markdown(balls_html(gaps.index.tolist()), unsafe_allow_html=True)
    st.caption(" · ".join(f"{n}번 {g}회째" for n, g in gaps.items()))

    feat = analyzer.features(df.tail(last_n) if last_n else df)
    actual_odd = feat["odd"].value_counts(normalize=True).reindex(range(7), fill_value=0)
    theory_odd = probability.odd_count_theory()
    fig = go.Figure([
        go.Bar(name="실제", x=[f"{k}:{6 - k}" for k in range(7)], y=actual_odd.values,
               marker_color="#69C8F2"),
        go.Bar(name="이론", x=[f"{k}:{6 - k}" for k in range(7)], y=theory_odd.values,
               marker_color="#AAAAAA"),
    ])
    fig.update_yaxes(tickformat=".0%", fixedrange=True)
    fig.update_xaxes(fixedrange=True)
    small_layout(fig, 260, "홀:짝 비율 — 실제 vs 이론")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    fig = go.Figure(go.Histogram(x=feat["sum"], nbinsx=30, marker_color="#FF7272"))
    fig.update_xaxes(title="6개 번호의 합", fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(small_layout(fig, 240, "번호 합계 분포"), width="stretch",
                    config=PLOT_CONFIG)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**구간별 출현**")
        st.dataframe(analyzer.section_counts(df, last_n).rename("횟수"), width="stretch")
    with c2:
        st.markdown("**함께 자주 나온 쌍**")
        st.dataframe(analyzer.top_pairs(df, 5, last_n), hide_index=True, width="stretch")
    st.caption(f"연번이 포함된 회차: {(feat['consec'] > 0).mean():.0%}")

    stat, p = analyzer.chi_square(df, last_n)
    st.markdown("**무작위성 검정 (카이제곱)**")
    if p > 0.05:
        st.info(f"p값 {p:.3f} — 번호별 출현 차이는 무작위 추첨에서 자연스럽게 생기는 수준입니다. "
                "‘많이 나온 번호’가 앞으로도 더 잘 나온다는 근거는 없습니다.")
    else:
        st.warning(f"p값 {p:.3f} — 이 기간에는 우연치고 큰 치우침이 보입니다. "
                   "다만 기간을 여러 번 바꿔 보면 이런 결과는 가끔 우연히도 나옵니다.")

# ---------------------------------------------------------------- 확률
with tab_prob:
    table = probability.prize_table()
    show = table.assign(확률=table["1/n"].map(lambda x: f"1 / {x:,.1f}"),
                        **{"경우의 수": table["경우의 수"].map("{:,}".format)})
    st.dataframe(show[["등수", "조건", "경우의 수", "확률"]], hide_index=True,
                 width="stretch")
    st.caption(f"전체 조합 수 C(45,6) = {probability.TOTAL:,}")

    tickets = st.number_input("매주 몇 장 사나요?", 1, 100, 5)
    rows = []
    for r in range(1, 6):
        weeks = probability.weeks_to_win(r, tickets)
        rows.append({"등수": f"{r}등",
                     "이번 주 당첨 확률": f"{probability.prob_with_tickets(r, tickets):.6%}",
                     "50% 확률로 한 번 당첨까지": f"약 {weeks:,.0f}주 ({weeks / 52:,.1f}년)"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    ev = probability.expected_return(df)
    st.metric("1,000원으로 기대할 수 있는 회수액 (최근 52회 평균, 세전)",
              won(ev["기대 회수액"]), f"회수율 {ev['회수율']:.0%}", delta_color="off")
    st.caption("판매액의 약 절반만 당첨금으로 돌아가는 구조라 장기적으로는 구매액의 절반 정도를 잃게 됩니다.")

# ---------------------------------------------------------------- 전략 검증
with tab_bt:
    st.write("과거 각 회차 **직전까지의 데이터만** 보고 번호를 만든 뒤, 실제 결과와 비교합니다.")
    n_rounds = st.slider("검증할 회차 수", 50, 500, 100, step=50)
    per_round = st.slider("회차당 게임 수", 1, 10, 5)
    if st.button("검증 시작", type="primary", width="stretch"):
        bar = st.progress(0.0, "계산 중…")
        st.session_state["bt"] = (backtest.run(df, n_rounds, per_round,
                                               progress=lambda x: bar.progress(x, "계산 중…")),
                                  n_rounds * per_round)
        bar.empty()
    if "bt" in st.session_state and "우연 범위 하한" not in st.session_state["bt"][0]:
        del st.session_state["bt"]  # 이전 버전 결과는 버리고 다시 계산하도록 함
    if "bt" in st.session_state:
        res, n = st.session_state["bt"]
        mean = backtest.MEAN
        lo, hi = res["우연 범위 하한"], res["우연 범위 상한"]
        inside_mask = res["판정"] == "우연 범위 안"
        fig = go.Figure()
        # 전략마다 다른 우연 범위 (회색 막대)
        fig.add_trace(go.Bar(y=res["전략"], x=hi - lo, base=lo, orientation="h",
                             marker_color="rgba(170,170,170,.35)", width=.8,
                             hovertemplate="우연 범위 %{base:.3f} ~ %{x:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            y=res["전략"], x=res["평균 일치 개수"], mode="markers+text",
            marker=dict(size=14, color=["#2B4C7E" if ok else "#FF7272" for ok in inside_mask]),
            text=res["평균 일치 개수"].map("{:.3f}".format), textposition="top center",
            hovertemplate="%{y}: %{x:.3f}개<extra></extra>"))
        fig.add_vline(x=mean, line_dash="dot")
        span = max(abs(lo - mean).max(), abs(hi - mean).max(),
                   abs(res["평균 일치 개수"] - mean).max()) + .05
        fig.update_xaxes(range=[mean - span, mean + span], fixedrange=True)
        fig.update_yaxes(autorange="reversed", fixedrange=True)
        small_layout(fig, 320, "평균 일치 개수")
        fig.update_layout(barmode="overlay")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption("점선 = 무작위 이론값 0.8개 · 회색 막대 = 그 전략이 우연히 낼 수 있는 범위(약 95%). "
                   "한 회차의 게임들이 같은 번호를 많이 공유하는 전략일수록 범위가 넓어집니다.")
        view = res.assign(**{"5등 이상 적중률": res["5등 이상 적중률"].map("{:.2%}".format),
                             "회수율": res["회수율"].map("{:.0%}".format)})
        st.dataframe(view, hide_index=True, width="stretch")
        outside = (~inside_mask).sum()
        if outside == 0:
            st.success("모든 전략이 우연 범위 안에 있습니다. 즉, 무작위로 고른 것과 성적 차이가 없습니다.")
        else:
            st.info(f"{outside}개 전략이 우연 범위를 벗어났습니다. 범위는 약 95% 기준이라, "
                    "전략 5개를 함께 시험하면 효과가 없어도 네 번에 한 번꼴로 하나쯤은 벗어납니다. "
                    "검증 회차를 바꿔 다시 돌려도 같은 방향으로 벗어나는지 확인해 보세요.")

# ---------------------------------------------------------------- 내 번호 조회
with tab_check:
    def _limit_six():
        """7개째를 누르면 그 선택은 취소하고 직전 6개를 유지합니다."""
        sel = st.session_state["mine"] or []
        if len(sel) > 6:
            st.session_state["mine"] = st.session_state.get("mine_prev", sel[:6])
            st.session_state["mine_over"] = True
        else:
            st.session_state["mine_prev"] = list(sel)

    def _clear_mine():
        st.session_state["mine"] = []
        st.session_state["mine_prev"] = []

    with st.container(key="ball_grid"):
        st.pills("번호 6개를 누르세요", list(range(1, 46)), selection_mode="multi",
                 key="mine", on_change=_limit_six)
    mine = sorted(st.session_state.get("mine") or [])
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.caption(f"{len(mine)}/6개 선택됨 · 다시 누르면 선택이 풀립니다")
    c2.button("모두 해제", on_click=_clear_mine, width="stretch", disabled=not mine)
    if st.session_state.pop("mine_over", False):
        st.toast("번호는 6개까지만 고를 수 있습니다. 다른 번호를 먼저 풀어 주세요.")
    if len(mine) == 6:
        st.markdown(balls_html(sorted(mine)), unsafe_allow_html=True)
        counts = {r: [] for r in range(1, 6)}
        total = 0
        for _, row in df.iterrows():
            _, r = backtest.rank_of(mine, row[NUM].astype(int).tolist(), int(row["bonus"]))
            if r:
                counts[r].append(int(row["round"]))
                total += int(row[f"win{r}_amount"])
        spent = len(df) * probability.TICKET_PRICE
        st.write(f"1회부터 {int(latest['round'])}회까지 매주 이 번호를 샀다면:")
        st.dataframe(pd.DataFrame([{"등수": f"{r}등", "당첨 횟수": len(v),
                                    "회차": ", ".join(map(str, v[-5:])) + (" …" if len(v) > 5 else "")}
                                   for r, v in counts.items()]),
                     hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.metric("쓴 돈", won(spent))
        c2.metric("받은 돈 (세전)", won(total), f"{total / spent:.0%}", delta_color="off")

st.divider()
st.caption("분석·학습용 프로그램입니다. 당첨을 보장하지 않으며, 구매는 감당할 수 있는 범위에서만 하세요. "
           "데이터 출처: 동행복권")
