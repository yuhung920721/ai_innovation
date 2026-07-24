# services/report_service.py
# ------------------------------------------------------------
# AI 復健報告服務
#
# 對外只需要用到一個函式：submit_session(...)
# 由各動作偵測程式（shoulder_*.py）在「剛完成」的那一瞬間呼叫。
#
# 內部流程：
#   1. 立刻把原始訓練數據寫入資料庫（同步、快速，不會卡住偵測迴圈）
#   2. 開一條背景執行緒去呼叫 Gemini API 產生 AI 建議報告
#   3. 報告產生完成後寫回資料庫，狀態從 pending 變成 ready
#
# 前端/Flask 路由只需要呼叫 db.get_report_by_session(session_id)
# 查詢報告目前的狀態與內容，不需要知道 Gemini 或背景執行緒的存在。
# ------------------------------------------------------------

import os
import threading
import traceback

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai

from services import db

load_dotenv()  # 讀取專案根目錄的 .env 檔案，把 GEMINI_API_KEY 載入環境變數

# ---- 初始化 ----
db.init_db()

GEMINI_MODEL = "gemini-2.5-flash"  # 如需使用更新的模型，請至 ai.google.dev 確認目前可用模型名稱

_client = None


def _get_client():
    """延遲初始化 Gemini client，避免沒有設定 API key 時一 import 就出錯"""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "找不到 GEMINI_API_KEY，請在環境變數或 .env 檔案中設定後再啟動系統"
            )
        _client = genai.Client(api_key=api_key)
    return _client


class RehabReport(BaseModel):
    """要求 Gemini 回傳的結構化報告格式，方便前端直接渲染，不需要自己解析文字"""
    summary: str            # 整體評語（2-3句）
    suggestions: list[str]  # 姿勢調整/改善建議（條列，3-5點）
    risk_note: str          # 風險提醒（若無明顯風險，簡短說明即可，不需誇大）
    next_session_advice: str  # 下次訓練的調整建議（例如次數、休息、強度）


SYSTEM_INSTRUCTION = (
    "你是一位物理治療輔助 AI，根據使用者本次上肢復健訓練的關節角度數據，"
    "給予專業但淺顯易懂的動作調整建議。"
    "請注意：不做醫療診斷、不使用『疑似XX症』等診斷用語，"
    "只針對動作標準度、對稱性、疲勞跡象等可從數據觀察到的現象提出建議。"
    "語氣需溫和、鼓勵，避免使用者感到挫折。"
)


def _build_prompt(session_data):
    rep_history = session_data["rep_history"]

    # 從逐次紀錄粗略判斷分數趨勢，讓 Gemini 不用自己做數學也能看出趨勢
    trend_note = "資料不足，無法判斷趨勢"
    if len(rep_history) >= 2:
        first_half = rep_history[: len(rep_history) // 2]
        second_half = rep_history[len(rep_history) // 2 :]
        avg_first = sum(r["rep_score"] for r in first_half) / len(first_half)
        avg_second = sum(r["rep_score"] for r in second_half) / len(second_half)
        if avg_second < avg_first - 8:
            trend_note = f"後段動作分數({avg_second:.1f})明顯低於前段({avg_first:.1f})，可能有疲勞跡象"
        elif avg_second > avg_first + 8:
            trend_note = f"後段動作分數({avg_second:.1f})高於前段({avg_first:.1f})，動作有隨訓練變熟練"
        else:
            trend_note = "全程分數表現穩定"

    return f"""
請根據以下這一次復健訓練的數據，產生一份復健建議報告：

動作名稱：{session_data['exercise_type']}
偵測手臂：{session_data['side']}
目標次數：{session_data['target_count']}
完成次數：{session_data['completed_count']}
平均動作標準度分數（0-100）：{session_data['average_score']:.1f}
本次訓練耗時（秒）：{session_data['duration_sec']:.0f}
逐次動作分數紀錄：{rep_history}
趨勢觀察：{trend_note}
"""


def _generate_ai_report(session_id):
    """背景執行緒目標函式：呼叫 Gemini API 並把結果寫回資料庫"""
    try:
        session_data = db.get_session(session_id)
        if session_data is None:
            raise ValueError(f"找不到 session_id={session_id} 的訓練紀錄")

        client = _get_client()
        prompt = _build_prompt(session_data)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_schema": RehabReport,
            },
        )
        report = RehabReport.model_validate_json(response.text)

        db.update_report_ready(
            session_id=session_id,
            summary=report.summary,
            suggestions=report.suggestions,
            risk_note=report.risk_note,
            next_session_advice=report.next_session_advice,
        )
        print(f"[report_service] AI 報告已產生完成 (session_id={session_id})")

    except Exception as e:
        # demo 現場最怕背景執行緒的例外被吞掉、前端一直卡在 pending 不知道為什麼
        # 這裡把完整錯誤訊息存進資料庫，方便除錯，也方便前端顯示 fallback 訊息
        error_detail = f"{type(e).__name__}: {e}"
        print(f"[report_service] 產生報告失敗 (session_id={session_id}): {error_detail}")
        traceback.print_exc()
        db.update_report_failed(session_id, error_detail)


def _handle_session(
    exercise_type,
    side,
    target_count,
    completed_count,
    average_score,
    duration_sec,
    rep_history,
    user_id,
):
    """
    背景執行緒真正做事的地方：寫入訓練紀錄、建立 pending 報告、呼叫 Gemini API。
    整段都在背景執行，呼叫端（動作偵測迴圈）完全不會被這裡面的任何一步卡住，
    包含資料庫寫入本身（雖然通常很快，但仍然是磁碟 I/O，不該留在畫面更新的執行緒裡）。
    """
    try:
        session_id = db.insert_session(
            user_id=user_id,
            exercise_type=exercise_type,
            side=side,
            target_count=target_count,
            completed_count=completed_count,
            average_score=average_score,
            duration_sec=duration_sec,
            rep_history=rep_history,
        )
        db.insert_pending_report(session_id)
        _generate_ai_report(session_id)
    except Exception as e:
        # 這裡出錯代表連「寫入訓練紀錄」都失敗了（比報告產生失敗更嚴重），
        # 印出來方便除錯，但不往外拋，避免影響呼叫端。
        print(f"[report_service] 處理訓練紀錄失敗: {type(e).__name__}: {e}")
        traceback.print_exc()


def submit_session(
    exercise_type,
    side="right",
    target_count=5,
    completed_count=0,
    average_score=0.0,
    duration_sec=0.0,
    rep_history=None,
    user_id=None,
):
    """
    公開 API：各動作偵測程式在「剛完成」的當下呼叫這個函式。

    這個函式本身「不做任何事」，只是把資料打包丟給背景執行緒去處理
    （包含寫入資料庫），然後立刻回傳。呼叫端（也就是攝影機畫面更新的
    主迴圈）幾乎不會感受到任何延遲，因為連磁碟寫入都不在主執行緒發生。

    注意：因為是 fire-and-forget，這裡不會回傳 session_id
    （要查詢報告狀態，請用 check_report.py 或之後前端的 API 查詢）。
    """
    thread = threading.Thread(
        target=_handle_session,
        args=(
            exercise_type,
            side,
            target_count,
            completed_count,
            average_score,
            duration_sec,
            rep_history or [],
            user_id,
        ),
        daemon=True,
    )
    thread.start()


def warm_up():
    """
    在程式啟動階段（開攝影機之前）呼叫一次，把 google-genai / pydantic 等
    套件的載入成本、以及 SQLite 建表的開銷，提前在「載入中」階段付掉，
    避免使用者完成動作的那一瞬間才第一次觸發這些成本，造成畫面卡頓。
    """
    try:
        _get_client()
    except RuntimeError:
        # 還沒設定 GEMINI_API_KEY 也沒關係，warm_up 的目的只是預先載入套件，
        # 真正的 API key 檢查留到實際呼叫 API 時再處理。
        pass
