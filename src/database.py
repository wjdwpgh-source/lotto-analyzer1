"""2단계: 데이터 저장·불러오기·검증 (SQLite + CSV 백업)."""
import sqlite3
from pathlib import Path

import pandas as pd

from . import collector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "lotto.db"
CSV_PATH = DATA_DIR / "lotto_history.csv"
NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def validate(df: pd.DataFrame) -> list:
    """데이터에 문제가 있으면 문제 목록을, 없으면 빈 목록을 돌려줍니다."""
    problems = []
    nums = df[NUM_COLS + ["bonus"]]
    if ((nums < 1) | (nums > 45)).any().any():
        problems.append("1~45 범위를 벗어난 번호가 있습니다.")
    dup = nums.apply(lambda r: len(set(r)) != 7, axis=1)
    if dup.any():
        problems.append(f"번호가 중복된 회차: {df.loc[dup, 'round'].tolist()[:10]}")
    missing = sorted(set(range(1, int(df["round"].max()) + 1)) - set(df["round"]))
    if missing:
        problems.append(f"누락된 회차 {len(missing)}개: {missing[:10]}")
    return problems


def save(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("draws", con, if_exists="replace", index=False)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def load() -> pd.DataFrame:
    """SQLite → CSV 순서로 불러옵니다. 둘 다 없으면 빈 표를 돌려줍니다."""
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            return pd.read_sql("SELECT * FROM draws ORDER BY round", con)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH).sort_values("round").reset_index(drop=True)
    return pd.DataFrame(columns=collector.COLUMNS)


def update():
    """최신 데이터를 받아 저장합니다. 실패하면 기존 데이터를 그대로 씁니다.
    반환값: (DataFrame, 안내 메시지)"""
    old = load()
    old_max = int(old["round"].max()) if len(old) else 0
    try:
        new = collector.fetch_all()
    except Exception as e:  # 네트워크 오류, 사이트 변경 등
        return old, f"업데이트에 실패해 저장된 데이터({old_max}회까지)를 사용합니다. 원인: {e}"
    problems = validate(new)
    if problems:
        return old, "받은 데이터에 문제가 있어 저장하지 않았습니다: " + " / ".join(problems)
    save(new)
    added = int(new["round"].max()) - old_max
    msg = f"{added}개 회차를 새로 받았습니다." if added > 0 else "이미 최신 데이터입니다."
    return new, msg
