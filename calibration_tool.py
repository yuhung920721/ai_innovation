# calibration_tool.py
# ------------------------------------------------------------
# 校正模式小工具
#
# 用途：不做次數計算，只負責把四個復健動作共用的關鍵角度/距離
#       同時即時顯示出來，讓你們錄影測試多次、記錄真實數據，
#       再回去調整 shoulder_abduction.py 等四支程式最上面的角度門檻。
#
# 重置機制（跟正式動作程式一致，並加上「停留確認」）：
#   每一「次」動作，都必須先回到「手臂自然垂下」的中立姿勢（角度接近0度），
#   才會開始被視為新的一次動作記錄；動作結束回到中立姿勢時，
#   自動印出這一次動作過程中出現的極值角度／距離。
#
#   為了避免手臂放下過程中「短暫晃過中立角度」就被誤判成動作已結束
#   （例如分兩段放下手臂、中途停頓一下），
#   回到中立姿勢必須「連續保持 NEUTRAL_HOLD_SECONDS 秒以上」才算數，
#   但「開始動作」的判定不需要等待，手一抬起就立刻反應。
#
# 節點顯示：
#   已改用 pose_utils.draw_body_pose()，只畫身體骨架，
#   不畫鼻子/眼睛/嘴巴等臉部表情節點（耳朵保留，因為後舉動作要用）。
#
# 操作方式：
#   python calibration_tool.py
#   r：清空目前記錄的所有次數
#   q 或 ESC：結束並印出統計摘要（建議的角度門檻）
# ------------------------------------------------------------

import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from pose_utils import (
    mp_pose,
    get_point,
    calculate_angle,
    euclidean_distance,
    LANDMARK_NAMES,
    OPPOSITE_SHOULDER_NAME,
    draw_body_pose,
)

# ---- 中立姿勢(重置)判斷門檻 ----
# 手臂抬舉角度(髖-肩-腕)低於這個值，視為回到「手臂自然垂下」的中立姿勢
NEUTRAL_ANGLE_MAX = 30
# 角度必須連續低於 NEUTRAL_ANGLE_MAX 多久，才算「真的」回到中立姿勢
# （避免放下手臂過程中短暫晃過門檻，就被誤判成一次動作已結束）
NEUTRAL_HOLD_SECONDS = 0.3
VISIBILITY_THRESHOLD = 0.5

fontpath = "NotoSansTC-Regular.ttf"
font_large = ImageFont.truetype(fontpath, 34)
font_small = ImageFont.truetype(fontpath, 24)


class CalibrationSession:
    """
    負責「回到中立姿勢才開始新一次記錄」的狀態機，
    並在每一次動作過程中，持續追蹤四個關鍵數值的極值：
      - lift_angle   : 髖-肩-腕的抬舉角度 (側舉/前舉用，最大值)
      - elbow_angle  : 肩-肘-腕的手肘彎曲角度 (平舉/後舉用，最小值＝最彎)
      - wrist_opp_shoulder_dist : 手腕到對側肩膀的距離 (平舉/手掌碰肩用，最小值)
      - wrist_ear_dist          : 手腕到耳朵的距離 (後舉/貼頸用，最小值)

    「開始動作」的判定是即時反應（手一抬起就立刻算開始），
    但「回到中立姿勢／動作結束」的判定，需要角度連續低於門檻
    達 NEUTRAL_HOLD_SECONDS 秒以上才算數，避免放下手臂過程中
    的短暫晃動被誤判成動作已經結束。
    """

    def __init__(self):
        self.state = "NEUTRAL_WAIT"  # NEUTRAL_WAIT -> READY -> IN_MOTION -> (回到NEUTRAL_WAIT)
        self.rep_count = 0
        self.records = []
        self.neutral_since = None  # 開始連續低於門檻的時間戳記
        self._reset_extremes()

    def _reset_extremes(self):
        self.max_lift_angle = -np.inf
        self.min_elbow_angle = np.inf
        self.min_wrist_opp_dist = np.inf
        self.min_wrist_ear_dist = np.inf

    def _confirmed_neutral(self, raw_neutral, now):
        """判斷是否「連續」低於中立門檻達到指定秒數"""
        if not raw_neutral:
            self.neutral_since = None
            return False
        if self.neutral_since is None:
            self.neutral_since = now
        return (now - self.neutral_since) >= NEUTRAL_HOLD_SECONDS

    def hold_progress(self, raw_neutral, now):
        """回傳 0~1 的停留確認進度，給畫面顯示用"""
        if not raw_neutral or self.neutral_since is None:
            return 0.0
        elapsed = now - self.neutral_since
        return min(elapsed / NEUTRAL_HOLD_SECONDS, 1.0)

    def update(self, lift_angle, elbow_angle, wrist_opp_dist, wrist_ear_dist, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        raw_neutral = lift_angle <= NEUTRAL_ANGLE_MAX
        confirmed_neutral = self._confirmed_neutral(raw_neutral, now)

        if self.state == "NEUTRAL_WAIT":
            if confirmed_neutral:
                self.state = "READY"

        elif self.state == "READY":
            if not raw_neutral:
                # 手一抬起就立刻視為開始新一次動作，不需要等待確認
                self._reset_extremes()
                self.state = "IN_MOTION"

        elif self.state == "IN_MOTION":
            # 動作進行中，持續更新極值
            self.max_lift_angle = max(self.max_lift_angle, lift_angle)
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.min_wrist_opp_dist = min(self.min_wrist_opp_dist, wrist_opp_dist)
            self.min_wrist_ear_dist = min(self.min_wrist_ear_dist, wrist_ear_dist)

            if confirmed_neutral:
                # 連續回到中立姿勢達門檻秒數，這一次動作記錄才真正結束
                self.rep_count += 1
                record = {
                    "rep": self.rep_count,
                    "max_lift_angle": self.max_lift_angle,
                    "min_elbow_angle": self.min_elbow_angle,
                    "min_wrist_opp_dist": self.min_wrist_opp_dist,
                    "min_wrist_ear_dist": self.min_wrist_ear_dist,
                }
                self.records.append(record)
                self._print_record(record)
                self.state = "NEUTRAL_WAIT"

    def _print_record(self, r):
        print("=" * 50)
        print(f"第 {r['rep']} 次動作記錄")
        print(f"  最大抬舉角度(側舉/前舉用)         : {r['max_lift_angle']:.1f}°")
        print(f"  最小手肘彎曲角度(平舉/後舉用)      : {r['min_elbow_angle']:.1f}°")
        print(f"  最小 手腕-對側肩 距離(平舉用)      : {r['min_wrist_opp_dist']:.3f}")
        print(f"  最小 手腕-耳朵 距離(後舉用)        : {r['min_wrist_ear_dist']:.3f}")
        print("=" * 50)

    def reset_all(self):
        self.rep_count = 0
        self.records = []
        self._reset_extremes()
        self.neutral_since = None
        self.state = "NEUTRAL_WAIT"
        print(">>> 已清空所有記錄，請重新開始 <<<")

    def print_summary(self):
        print("\n" + "#" * 50)
        print(f"校正結束，共記錄 {len(self.records)} 次動作")
        if not self.records:
            print("沒有任何記錄，無法給出建議門檻。")
            print("#" * 50)
            return

        lift_vals = [r["max_lift_angle"] for r in self.records]
        elbow_vals = [r["min_elbow_angle"] for r in self.records]
        opp_vals = [r["min_wrist_opp_dist"] for r in self.records]
        ear_vals = [r["min_wrist_ear_dist"] for r in self.records]

        def stat_line(name, vals):
            print(f"{name}: 最小={min(vals):.2f}  最大={max(vals):.2f}  平均={np.mean(vals):.2f}")

        stat_line("最大抬舉角度 (lift_angle)", lift_vals)
        stat_line("最小手肘角度 (elbow_angle)", elbow_vals)
        stat_line("最小 腕-對側肩 距離 (wrist_opp_dist)", opp_vals)
        stat_line("最小 腕-耳 距離 (wrist_ear_dist)", ear_vals)

        avg_lift = float(np.mean(lift_vals))
        avg_elbow = float(np.mean(elbow_vals))
        avg_opp = float(np.mean(opp_vals))
        avg_ear = float(np.mean(ear_vals))

        print("\n--- 建議填入各程式的角度／距離門檻（平均值 ± 緩衝，僅供參考） ---")
        print(f"[手臂側舉/前舉]  S1(中間姿勢) 約 90°附近；S2(舉高) 建議 S2_ANGLE_MIN = {max(avg_lift - 20, 100):.0f}")
        print(f"[手臂平舉]       S2(打直平舉) 建議 ELBOW_STRAIGHT_MIN = {min(avg_elbow + 30, 170):.0f} "
              f"(若此值偏低請改看你自己平舉打直時的 elbow_angle 實測值)")
        print(f"[手臂平舉/後舉]  S1(碰肩/貼頸) 建議距離門檻 = {avg_opp + 0.05:.2f} / {avg_ear + 0.05:.2f} "
              f"(比實測平均值再放寬一點，避免太嚴格判定不到)")
        print("#" * 50)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 900, 650)

    session = CalibrationSession()

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            now = time.time()

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            lift_angle = 0.0
            elbow_angle = 180.0
            wrist_opp_dist = 999.0
            wrist_ear_dist = 999.0
            landmarks_ok = False

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                hip, hip_vis = get_point(lm, LANDMARK_NAMES["HIP"])
                shoulder, sh_vis = get_point(lm, LANDMARK_NAMES["SHOULDER"])
                elbow, el_vis = get_point(lm, LANDMARK_NAMES["ELBOW"])
                wrist, wr_vis = get_point(lm, LANDMARK_NAMES["WRIST"])
                ear, ear_vis = get_point(lm, LANDMARK_NAMES["EAR"])
                opp_shoulder, opp_vis = get_point(lm, OPPOSITE_SHOULDER_NAME)

                if min(hip_vis, sh_vis, el_vis, wr_vis, ear_vis, opp_vis) >= VISIBILITY_THRESHOLD:
                    landmarks_ok = True
                    lift_angle = calculate_angle(hip, shoulder, wrist)
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    wrist_opp_dist = euclidean_distance(wrist[:2], opp_shoulder[:2])
                    wrist_ear_dist = euclidean_distance(wrist[:2], ear[:2])

                    session.update(lift_angle, elbow_angle, wrist_opp_dist, wrist_ear_dist, timestamp=now)

                # 只畫身體骨架，排除臉部表情節點
                draw_body_pose(image, results.pose_landmarks)

            # ---- 畫面文字疊加 ----
            image = cv2.flip(image, 1)
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            state_text = {
                "NEUTRAL_WAIT": "請回到中立姿勢(手臂自然垂下)",
                "READY": "準備就緒，請開始動作",
                "IN_MOTION": "動作記錄中...",
            }.get(session.state, session.state)

            state_color = (0, 0, 255) if session.state == "NEUTRAL_WAIT" else (
                (0, 150, 255) if session.state == "READY" else (0, 200, 0)
            )

            draw.text((10, 10), f"狀態：{state_text}", fill=state_color, font=font_large)
            draw.text((10, 55), f"已記錄次數：{session.rep_count}", fill=(0, 0, 0), font=font_large)

            # 顯示「回到中立姿勢」的停留確認進度（僅在角度已經低於門檻、正在計時確認時顯示）
            raw_neutral = landmarks_ok and (lift_angle <= NEUTRAL_ANGLE_MAX)
            progress = session.hold_progress(raw_neutral, now)
            if 0 < progress < 1.0 and session.state == "IN_MOTION":
                draw.text(
                    (10, 240),
                    f"確認回到中立姿勢中... {progress * 100:.0f}%",
                    fill=(255, 140, 0),
                    font=font_small,
                )

            if not landmarks_ok:
                draw.text((10, 100), "偵測不到完整關節點，請確認全身/半身入鏡", fill=(255, 0, 0), font=font_small)
            else:
                draw.text((10, 100), f"抬舉角度 lift_angle   : {lift_angle:6.1f}°", fill=(0, 0, 0), font=font_small)
                draw.text((10, 130), f"手肘角度 elbow_angle  : {elbow_angle:6.1f}°", fill=(0, 0, 0), font=font_small)
                draw.text((10, 160), f"腕-對側肩距離         : {wrist_opp_dist:6.3f}", fill=(0, 0, 0), font=font_small)
                draw.text((10, 190), f"腕-耳距離             : {wrist_ear_dist:6.3f}", fill=(0, 0, 0), font=font_small)

            draw.text(
                (10, pil_image.height - 40),
                "按 r 清空記錄　按 q / ESC 結束並印出建議門檻",
                fill=(100, 100, 100),
                font=font_small,
            )

            image = np.array(pil_image)
            cv2.imshow("Calibration", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' 或 ESC
                break
            elif key == ord("r"):
                session.reset_all()

    cap.release()
    cv2.destroyAllWindows()

    session.print_summary()


if __name__ == "__main__":
    main()