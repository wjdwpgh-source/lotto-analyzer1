"""1단계: 동행복권에서 로또 6/45 역대 당첨번호를 수집합니다.

동행복권은 2025년 말 사이트를 개편하면서 예전 주소
(common.do?method=getLottoNumber)를 막았습니다.
현재는 아래 JSON 경로를 사용합니다. 공식 공개 API가 아니므로
나중에 바뀔 수 있으며, 그럴 땐 data/lotto_history.csv 백업으로 동작합니다.
"""
import requests
import pandas as pd

API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"),
    "Referer": "https://www.dhlottery.co.kr/lt645/result",
    "X-Requested-With": "XMLHttpRequest",
}

COLUMNS = ["round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
           "win1_count", "win1_amount", "win2_count", "win2_amount",
           "win3_count", "win3_amount", "win4_count", "win4_amount",
           "win5_count", "win5_amount", "sales"]


def _to_row(item: dict) -> dict:
    nums = sorted(int(item[f"tm{i}WnNo"]) for i in range(1, 7))
    d = str(item["ltRflYmd"])
    row = {"round": int(item["ltEpsd"]), "date": f"{d[:4]}-{d[4:6]}-{d[6:]}"}
    row.update({f"n{i + 1}": n for i, n in enumerate(nums)})
    row["bonus"] = int(item["bnsWnNo"])
    for r in range(1, 6):
        row[f"win{r}_count"] = int(item.get(f"rnk{r}WnNope") or 0)
        row[f"win{r}_amount"] = int(item.get(f"rnk{r}WnAmt") or 0)
    row["sales"] = int(item.get("rlvtEpsdSumNtslAmt") or 0)
    return row


def _request(param: str, timeout: int = 30) -> list:
    res = requests.get(API_URL, params={"srchLtEpsd": param},
                       headers=HEADERS, timeout=timeout)
    res.raise_for_status()
    if "json" not in res.headers.get("Content-Type", ""):
        raise RuntimeError("동행복권 응답이 JSON이 아닙니다. 조회 경로가 바뀌었을 수 있습니다.")
    return res.json()["data"]["list"]


def fetch_all() -> pd.DataFrame:
    """1회차부터 최신 회차까지 한 번에 가져옵니다."""
    items = _request("all")
    df = pd.DataFrame([_to_row(it) for it in items], columns=COLUMNS)
    return df.sort_values("round").reset_index(drop=True)


def fetch_round(round_no: int) -> dict:
    """특정 회차 1개만 가져옵니다."""
    items = _request(str(round_no), timeout=15)
    if not items:
        raise ValueError(f"{round_no}회차 데이터가 없습니다.")
    return _to_row(items[0])


if __name__ == "__main__":
    df = fetch_all()
    print(f"{len(df)}개 회차 수집 완료 (최신 {df['round'].max()}회, {df['date'].iloc[-1]})")
