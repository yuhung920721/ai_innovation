# check_report.py
# ------------------------------------------------------------
# 開發階段的檢查工具：不用等組員的資料庫/前端做好，
# 自己就能確認「動作完成 -> AI報告生成」這條路徑有沒有正常運作。
#
# 用法：
#   python check_report.py          # 列出最近 5 筆訓練紀錄與報告狀態
#   python check_report.py 3        # 只看 session_id = 3 的完整報告內容
# ------------------------------------------------------------

import sys
from services import db


def show_latest(limit=5):
    conn = db._get_conn()
    try:
        rows = conn.execute(
            "SELECT id, exercise_type, completed_count, average_score, created_at "
            "FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("目前資料庫裡還沒有任何訓練紀錄。請先完成一次動作偵測程式。")
        return

    print(f"{'session_id':<10} {'動作':<24} {'完成/目標':<10} {'平均分數':<8} {'報告狀態':<8} 建立時間")
    print("-" * 90)
    for row in rows:
        report = db.get_report_by_session(row["id"])
        status = report["status"] if report else "(無報告紀錄)"
        print(
            f"{row['id']:<10} {row['exercise_type']:<24} "
            f"{row['completed_count']:<10} {row['average_score']:<8.1f} {status:<8} {row['created_at']}"
        )


def show_detail(session_id):
    session_data = db.get_session(session_id)
    report = db.get_report_by_session(session_id)

    if session_data is None:
        print(f"找不到 session_id={session_id} 的訓練紀錄")
        return

    print("=== 訓練紀錄 ===")
    for k, v in session_data.items():
        print(f"{k}: {v}")

    print("\n=== AI 報告 ===")
    if report is None:
        print("(尚未建立報告紀錄，理論上不該發生)")
    elif report["status"] == "pending":
        print("狀態：產生中，請稍後再查詢一次")
    elif report["status"] == "failed":
        print(f"狀態：失敗\n錯誤訊息：{report['error_message']}")
        print("常見原因：GEMINI_API_KEY 沒設定、額度用完、或網路無法連線")
    else:
        print(f"整體評語：{report['summary']}")
        print(f"建議：{report['suggestions']}")
        print(f"風險提醒：{report['risk_note']}")
        print(f"下次訓練建議：{report['next_session_advice']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_detail(int(sys.argv[1]))
    else:
        show_latest()
