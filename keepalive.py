"""Streamlit Community Cloud 앱이 잠들지 않도록 접속하고, 잠들어 있으면 깨웁니다."""
import re
import sys
from playwright.sync_api import sync_playwright

URL = "https://jeho-lotto.streamlit.app/"
WAKE = re.compile(r"get this app back up", re.I)


def app_text(page):
    """앱 본문(iframe 안쪽)에서 '로또분석기' 제목이 보이는지 확인."""
    for f in page.frames:
        try:
            if f.get_by_text("로또분석기").count():
                return True
        except Exception:
            pass
    return False


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(10_000)

    clicked = False
    for f in page.frames:
        btn = f.get_by_role("button", name=WAKE)
        if btn.count():
            print("앱이 잠들어 있어 깨우기 버튼을 누릅니다.")
            btn.first.click()
            clicked = True
            break

    # 앱 화면이 뜰 때까지 최대 3분 대기
    for _ in range(36):
        if app_text(page):
            print("앱 정상 동작 확인" + (" (깨움)" if clicked else " (이미 깨어 있음)"))
            browser.close()
            sys.exit(0)
        page.wait_for_timeout(5_000)

    print("앱 화면을 확인하지 못했습니다.")
    page.screenshot(path="result.png", full_page=True)
    browser.close()
    sys.exit(1)
