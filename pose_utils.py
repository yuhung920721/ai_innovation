# pose_utils.py
# ------------------------------------------------------------
# 共用模組：MediaPipe Pose 初始化、關節座標擷取、角度計算
# 供 shoulder_abduction.py / shoulder_flexion.py /
#    shoulder_horizontal_adduction.py / shoulder_extension_stretch.py /
#    calibration_tool.py 共用
# ------------------------------------------------------------

import cv2
import time
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 使用哪一側手臂進行判斷（可依需求切換 'RIGHT' 或 'LEFT'）
# 注意：這裡的 LEFT/RIGHT 是「使用者的左右」，不是畫面上的左右
WORKING_SIDE = "RIGHT"

LANDMARK_NAMES = {
    "SHOULDER": f"{WORKING_SIDE}_SHOULDER",
    "ELBOW": f"{WORKING_SIDE}_ELBOW",
    "WRIST": f"{WORKING_SIDE}_WRIST",
    "HIP": f"{WORKING_SIDE}_HIP",
    "EAR": f"{WORKING_SIDE}_EAR",
}
OPPOSITE_SIDE = "LEFT" if WORKING_SIDE == "RIGHT" else "RIGHT"
OPPOSITE_SHOULDER_NAME = f"{OPPOSITE_SIDE}_SHOULDER"

# ---- 臉部節點設定（繪製骨架時排除，減少畫面雜訊） ----
# 0:鼻子 1-3:左眼(內/中/外) 4-6:右眼(內/中/外) 9-10:嘴巴(左/右)
# 保留 7,8(左右耳)，因為「手臂後舉」動作需要用耳朵座標判斷手是否貼近後頸
FACE_EXCLUDE_INDICES = {0, 1, 2, 3, 4, 5, 6, 9, 10}


def _build_body_connections():
    """從 MediaPipe 預設的 POSE_CONNECTIONS 中，過濾掉任何一端屬於臉部節點的連線"""
    return frozenset(
        conn
        for conn in mp_pose.POSE_CONNECTIONS
        if conn[0] not in FACE_EXCLUDE_INDICES and conn[1] not in FACE_EXCLUDE_INDICES
    )


BODY_POSE_CONNECTIONS = _build_body_connections()


def draw_body_pose(image, pose_landmarks, visibility_threshold=0.3):
    """
    只繪製身體骨架（肩、肘、腕、髖、膝、踝等）與連線，
    排除鼻子/眼睛/嘴巴等臉部表情節點，但保留耳朵節點。
    直接在傳入的 image (numpy array, BGR) 上繪製。
    """
    h, w = image.shape[:2]
    landmarks = pose_landmarks.landmark

    # 先畫連線
    for start_idx, end_idx in BODY_POSE_CONNECTIONS:
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        if start_lm.visibility < visibility_threshold or end_lm.visibility < visibility_threshold:
            continue
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(image, start_point, end_point, (255, 255, 255), 2)

    # 再畫節點（圓點）
    for idx, lm in enumerate(landmarks):
        if idx in FACE_EXCLUDE_INDICES:
            continue
        if lm.visibility < visibility_threshold:
            continue
        center = (int(lm.x * w), int(lm.y * h))
        cv2.circle(image, center, 4, (0, 255, 0), -1)

    return image


def get_point(landmarks, name):
    """
    從 MediaPipe 偵測結果中取出指定關節點的 (x, y, z) 座標與可視度(visibility)。
    landmarks: results.pose_landmarks.landmark (正規化座標 0~1)
               或 results.pose_world_landmarks.landmark (公尺為單位的3D世界座標)
    name: 例如 'RIGHT_SHOULDER'
    回傳: (x, y, z), visibility
    """
    idx = mp_pose.PoseLandmark[name].value
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z]), lm.visibility


def calculate_angle(a, b, c):
    """
    計算以 b 為頂點，a-b-c 三點所形成的夾角（單位：度）。
    a, b, c 皆為 numpy array，可為 2D (x, y) 或 3D (x, y, z)。
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def euclidean_distance(a, b):
    """計算兩點間的歐式距離（正規化座標尺度）"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.linalg.norm(a - b)


class RepCounter:
    """
    可重複使用的「動作計數器」狀態機，採嚴格依序流程：

        NEUTRAL_WAIT（中立姿勢，須連續保持 neutral_hold_seconds 秒以上）
            ↓ 確認回到中立姿勢
        AWAITING_S1（畫面提示：請做 S1 動作）
            ↓ 偵測到 S1 → 立即亮綠燈、立即前進到下一階段
        AWAITING_S2（畫面提示：請做 S2 動作）
            ↓ 偵測到 S2 → 立即亮綠燈、立即次數 +1
        NEUTRAL_WAIT（必須重新回到中立姿勢，才能開始下一次）

    設計重點（畫面回饋 vs. 分數精算 分離）：
      - 燈號亮起、次數累加，都是「即時反應」，一偵測到就立刻顯示，
        不會有任何延遲，符合一般使用者(尤其年長者)習慣的紅綠燈式回饋。
      - 分數的精算則在背景默默進行：即使已經亮燈、已經計數，
        只要使用者還持續停留在該姿勢附近微調，系統會持續追蹤
        出現過的最佳角度分數，直到真正確認離開該姿勢，才把
        這一次動作的最終分數定案。使用者不會感覺到任何等待，
        因為次數與燈號早就即時顯示了，只有「總得分」的計算
        在背後多花一點時間去抓最佳值。
    """

    def __init__(self, target_count=5, neutral_hold_seconds=0.3, exit_hold_seconds=0.15):
        self.phase = "NEUTRAL_WAIT"  # NEUTRAL_WAIT -> AWAITING_S1 -> AWAITING_S2 -> (計數後)NEUTRAL_WAIT
        self.neutral_since = None
        self.s1_done = False
        self.s2_done = False
        self.total = 0
        self.target_count = target_count
        self.target_percentage = 0.0
        self.score_sum = 0.0
        self.finished = False
        self.neutral_hold_seconds = neutral_hold_seconds
        self.exit_hold_seconds = exit_hold_seconds

        # 背景分數精算用的追蹤變數
        self._s1_best = None
        self._s2_best = None
        self._s2_exit_since = None  # 用來確認「真的離開S2」，才把分數定案
        self._score_pending = False  # 這一次動作已經計數，但分數尚未定案

    def _confirmed_neutral(self, is_neutral, now):
        """判斷是否「連續」處於中立姿勢達到指定秒數，避免短暫晃動被誤判"""
        if not is_neutral:
            self.neutral_since = None
            return False
        if self.neutral_since is None:
            self.neutral_since = now
        return (now - self.neutral_since) >= self.neutral_hold_seconds

    def hold_progress(self, is_neutral, now):
        """回傳 0~1 的「回到中立姿勢」確認進度，供畫面顯示用"""
        if not is_neutral or self.neutral_since is None:
            return 0.0
        elapsed = now - self.neutral_since
        return min(elapsed / self.neutral_hold_seconds, 1.0)

    def _finalize_pending_score(self):
        """把目前尚未定案的分數，用背景追蹤到的最佳值計入總分"""
        if self._score_pending and self._s1_best is not None and self._s2_best is not None:
            rep_score = (self._s1_best + self._s2_best) / 2.0
            self.score_sum += rep_score
        self._score_pending = False

    def _reset_for_new_rep(self):
        self.s1_done = False
        self.s2_done = False
        self._s1_best = None
        self._s2_best = None
        self._s2_exit_since = None
        self.phase = "AWAITING_S1"

    def update(self, is_s1, is_s2, is_neutral, confidence_this_frame=100.0, timestamp=None):
        """
        is_s1 / is_s2: 布林值，代表本影格是否偵測到達成該子動作的角度條件
        is_neutral: 布林值，代表本影格是否處於中立姿勢(S0，例如手臂自然垂下)
        confidence_this_frame: 本影格的「動作標準度」分數(0~100)，
                                由角度與目標角度的接近程度換算而來
        timestamp: 目前時間(time.time())，未提供則自動取用系統時間
        """
        if self.finished:
            return

        now = timestamp if timestamp is not None else time.time()
        confirmed_neutral = self._confirmed_neutral(is_neutral, now)

        if self.phase == "NEUTRAL_WAIT":
            # 尚未確認回到中立姿勢前，不接受任何 S1/S2 判定
            if confirmed_neutral:
                self._finalize_pending_score()
                self._reset_for_new_rep()
            return

        if self.total >= self.target_count:
            self._finalize_pending_score()
            self.finished = True
            return

        if self.phase in ("AWAITING_S1", "AWAITING_S2") and confirmed_neutral:
            # 使用者中途放棄、完全回到中立姿勢卻還沒完成這一次動作
            # → 先把已經計數的部分定案，然後重新開始這一次動作
            self._finalize_pending_score()
            self._reset_for_new_rep()
            return

        if self.phase == "AWAITING_S1":
            if is_s1:
                # 立即亮燈、立即前進到下一階段（不延遲）
                self.s1_done = True
                self._s1_best = confidence_this_frame if self._s1_best is None else max(
                    self._s1_best, confidence_this_frame
                )
                self.phase = "AWAITING_S2"
            return

        if self.phase == "AWAITING_S2":
            # S1 可能還在附近微調姿勢，背景持續精進 S1 分數，不影響流程與畫面
            if is_s1:
                self._s1_best = max(self._s1_best, confidence_this_frame)

            if is_s2:
                if not self.s2_done:
                    # 第一次偵測到 S2 → 立即亮燈、立即次數 +1（不延遲）
                    self.s2_done = True
                    self.total += 1
                    self.target_percentage = (self.total / self.target_count) * 100
                    self._score_pending = True
                self._s2_best = confidence_this_frame if self._s2_best is None else max(
                    self._s2_best, confidence_this_frame
                )
                self._s2_exit_since = None
            elif self._score_pending:
                # 已經計數過，現在偵測不到S2了(使用者開始收手)
                # 用短暫的「離開確認」避免邊緣雜訊，確認後才把分數背景定案
                if self._s2_exit_since is None:
                    self._s2_exit_since = now
                if now - self._s2_exit_since >= self.exit_hold_seconds:
                    self._finalize_pending_score()
                    self.phase = "NEUTRAL_WAIT"
                    self.neutral_since = None

    @property
    def average_score(self):
        if self.total == 0:
            return 0.0
        return self.score_sum / self.total
