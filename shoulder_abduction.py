# shoulder_abduction.py
# ------------------------------------------------------------
# 動作名稱：手臂側舉（肩關節外展 Shoulder Abduction）
# 原程式檔名：Updown.py
#
# 【關節點】RIGHT_HIP - RIGHT_SHOULDER - RIGHT_ELBOW
# 【判斷邏輯】以髖-肩-肘所形成的夾角，代表手臂離開身體軀幹的外展角度
#             angle ≈ 0°   手臂自然下垂（S0 無動作）
#             angle ≈ 90°  手臂側平舉（S1 肩部側舉）
#             angle ≈ 150°~180°  手臂舉高貼近耳朵（S2 肩部上舉）
# ------------------------------------------------------------

import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from pose_utils import (
    mp_pose,
    get_point,
    calculate_angle,
    RepCounter,
    LANDMARK_NAMES,
    draw_body_pose,
)

# ---- 角度區間設定（可依實測數據微調） ----
S1_ANGLE_MIN, S1_ANGLE_MAX = 70, 110     # 側舉：約90度
S2_ANGLE_MIN = 150                        # 上舉：接近180度
S0_ANGLE_MAX = 20                         # 預備姿勢：手臂垂下
VISIBILITY_THRESHOLD = 0.5                # 關節點可信度門檻

TARGET_COUNT = 5

fontpath = "NotoSansTC-Regular.ttf"
font = ImageFont.truetype(fontpath, 30)


def classify_state(hip, shoulder, elbow):
    """
    根據髖-肩-肘夾角，回傳目前動作狀態: 'S0' / 'S1' / 'S2' / None(角度介於中間，判定不明確)
    """
    angle = calculate_angle(hip, shoulder, elbow)
    if angle <= S0_ANGLE_MAX:
        return "S0", angle
    elif S1_ANGLE_MIN <= angle <= S1_ANGLE_MAX:
        return "S1", angle
    elif angle >= S2_ANGLE_MIN:
        return "S2", angle
    else:
        return None, angle


def angle_to_confidence(angle, target_angle, tolerance=30):
    """
    將角度與目標角度的差距轉換成 0~100 的「動作標準度」分數，
    取代原本模型 softmax 信心值，數值越接近目標角度分數越高。
    """
    diff = abs(angle - target_angle)
    score = max(0.0, 100.0 * (1 - diff / tolerance))
    return score


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 800, 600)

    counter = RepCounter(target_count=TARGET_COUNT)
    action_mapping = {"S0": "無動作", "S1": "肩部側舉", "S2": "肩部上舉", None: "動作中"}

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
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

            predicted_label = "S0"
            angle_value = 0.0
            confidence = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                hip, hip_vis = get_point(lm, LANDMARK_NAMES["HIP"])
                shoulder, sh_vis = get_point(lm, LANDMARK_NAMES["SHOULDER"])
                elbow, el_vis = get_point(lm, LANDMARK_NAMES["ELBOW"])

                if min(hip_vis, sh_vis, el_vis) >= VISIBILITY_THRESHOLD:
                    state, angle_value = classify_state(hip, shoulder, elbow)
                    predicted_label = state if state else "S0"

                    if predicted_label == "S1":
                        confidence = angle_to_confidence(angle_value, 90)
                    elif predicted_label == "S2":
                        confidence = angle_to_confidence(angle_value, 180, tolerance=40)

                draw_body_pose(image, results.pose_landmarks)

            is_s1 = predicted_label == "S1"
            is_s2 = predicted_label == "S2"
            is_neutral = predicted_label == "S0"
            counter.update(is_s1, is_s2, is_neutral, confidence_this_frame=confidence, timestamp=now)

            # ---- 影像疊加顯示 ----
            image = cv2.flip(image, 1)
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            predicted_label_chinese = action_mapping.get(predicted_label, "動作中")
            draw.text((10, 10), f"目前動作：{predicted_label_chinese}", fill=(0, 0, 0), font=font)

            prompt_text = {
                "NEUTRAL_WAIT": "請回到中立姿勢(手臂自然垂下)",
                "AWAITING_S1": f"請做「{action_mapping['S1']}」動作",
                "AWAITING_S2": f"請做「{action_mapping['S2']}」動作",
            }.get(counter.phase, "")
            draw.text((10, 46), f"提示：{prompt_text}", fill=(0, 100, 200), font=font)

            draw.text((10, 82), f"角度：{angle_value:.1f}°", fill=(0, 0, 0), font=font)
            draw.text((10, 118), f"次數: {counter.total}", fill=(0, 0, 0), font=font)

            circle_color_s1 = (0, 255, 0) if counter.s1_done else (0, 0, 255)
            circle_color_s2 = (0, 255, 0) if counter.s2_done else (0, 0, 255)
            draw.text((450, 10), "側舉完成：", fill=(0, 0, 0), font=font)
            draw.ellipse([(600, 15), (620, 35)], fill=circle_color_s1)
            draw.text((450, 46), "上舉完成：", fill=(0, 0, 0), font=font)
            draw.ellipse([(600, 51), (620, 71)], fill=circle_color_s2)

            progress_bar_width = int((counter.target_percentage / 100) * pil_image.width)
            draw.rectangle(
                [(0, pil_image.height - 30), (progress_bar_width, pil_image.height)],
                fill=(138, 217, 255),
            )
            draw.text(
                (10, pil_image.height - 65),
                f"完成進度：{counter.target_percentage:.2f}%",
                fill=(0, 0, 0),
                font=font,
            )

            if counter.finished:
                draw.text(
                    (10, pil_image.height - 100),
                    f"總得分：{counter.average_score:.1f}分",
                    fill=(255, 0, 0),
                    font=font,
                )

            image = np.array(pil_image)
            cv2.imshow("Frame", image)

            if cv2.waitKey(1) != -1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
