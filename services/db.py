# services/db.py
# ------------------------------------------------------------
# 資料庫存取模組（SQLite）
# 管理兩張表：
#   sessions    -> 每一次完成的訓練紀錄（原始數值數據）
#   ai_reports  -> 對應每筆 session 產生的 AI 復健建議報告
#
# 使用 SQLite 是因為專題/demo 階段不需要額外架設資料庫服務，
# 檔案型資料庫即可運作；未來若要換成 MySQL/Postgres，
# 只需要修改這支檔案裡的連線方式與 SQL 語法，
# 不用動到 report_service.py 或任何動作偵測程式碼。
# ------------------------------------------------------------

import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rehab.db")


def _get_conn():
    """
    每次呼叫都開一個新連線再關閉，避免 SQLite 在多執行緒
    （背景執行緒呼叫 Gemini API）下的 thread-safety 問題。
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建立資料表（若不存在）。在 report_service 模組載入時會自動呼叫一次。"""
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT,
                exercise_type   TEXT NOT NULL,
                side            TEXT,
                target_count    INTEGER,
                completed_count INTEGER,
                average_score   REAL,
                duration_sec    REAL,
                rep_history     TEXT,   -- JSON 字串：每次動作的細節分數
                created_at      TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_reports (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    INTEGER NOT NULL REFERENCES sessions(id),
                status        TEXT NOT NULL DEFAULT 'pending',  -- pending / ready / failed
                summary       TEXT,
                suggestions   TEXT,   -- JSON 字串：list[str]
                risk_note     TEXT,
                next_session_advice TEXT,
                error_message TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_session(
    user_id,
    exercise_type,
    side,
    target_count,
    completed_count,
    average_score,
    duration_sec,
    rep_history,
):
    """寫入一筆訓練紀錄，回傳 session_id"""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO sessions
                (user_id, exercise_type, side, target_count, completed_count,
                 average_score, duration_sec, rep_history, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                exercise_type,
                side,
                target_count,
                completed_count,
                average_score,
                duration_sec,
                json.dumps(rep_history, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_session(session_id):
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["rep_history"] = json.loads(data["rep_history"] or "[]")
        return data
    finally:
        conn.close()


def insert_pending_report(session_id):
    """建立一筆狀態為 pending 的報告佔位紀錄，回傳 report id"""
    conn = _get_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO ai_reports (session_id, status, created_at, updated_at)
            VALUES (?, 'pending', ?, ?)
            """,
            (session_id, now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_report_ready(session_id, summary, suggestions, risk_note, next_session_advice):
    conn = _get_conn()
    try:
        conn.execute(
            """
            UPDATE ai_reports
            SET status = 'ready',
                summary = ?,
                suggestions = ?,
                risk_note = ?,
                next_session_advice = ?,
                updated_at = ?
            WHERE session_id = ?
            """,
            (
                summary,
                json.dumps(suggestions, ensure_ascii=False),
                risk_note,
                next_session_advice,
                datetime.now(timezone.utc).isoformat(),
                session_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_report_failed(session_id, error_message):
    conn = _get_conn()
    try:
        conn.execute(
            """
            UPDATE ai_reports
            SET status = 'failed', error_message = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (error_message, datetime.now(timezone.utc).isoformat(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_report_by_session(session_id):
    """
    給 Flask 前端路由用：查詢某個 session 對應的 AI 報告。
    回傳 None 表示這個 session_id 還沒有報告紀錄（理論上不該發生，
    因為 submit_session 會同時建立 pending 報告）。
    status 可能是 'pending' / 'ready' / 'failed'，前端依此顯示不同畫面。
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM ai_reports WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        if data.get("suggestions"):
            data["suggestions"] = json.loads(data["suggestions"])
        return data
    finally:
        conn.close()
