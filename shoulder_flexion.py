# shoulder_flexion.py
# ------------------------------------------------------------
# 動作名稱：手臂前舉（肩關節前屈 Shoulder Flexion）
# 原程式檔名：Wood.py
#
# 【關節點】RIGHT_HIP - RIGHT_SHOULDER - RIGHT_WRIST（軀幹到手腕的抬舉角度）
#           另外用 LEFT_WRIST 與 RIGHT_WRIST 的距離，確認雙手是否輕握在一起
# 【判斷邏輯】
# 【判斷邏輯】
#     angle ≈ 0°~35°(坐姿)  雙手垂放/放於大腿（S0 準備動作）
#     angle ≈ 70°           雙手前舉（S1 雙臂前舉）
#     angle ≈ 170°          雙手舉高貼耳（S2 雙臂上舉）
#     並要求 wrists_together=True，避免與「側舉」動作混淆(S0判斷除外)
#     （前舉時雙手應維持在身體中線附近，側舉時雙手會分開在身體兩側）
# ------------------------------------------------------------

import argparse
import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from pose_utils import (
    mp_pose,
    get_point,
    calculate_angle,
    euclidean_distance,
    RepCounter,
    build_landmark_names,
    draw_body_pose,
)

S1_ANGLE_MIN, S1_ANGLE_MAX = 60, 100      # 前舉：實測目標約80度
S2_ANGLE_MIN = 155                        # 上舉：實測目標約170度
S0_ANGLE_MAX = 50                         # 準備動作：坐姿時手放大腿上，實測角度門檻
WRISTS_TOGETHER_MAX_DIST = 0.15   # 正規化座標下，雙手腕距離門檻(依畫面比例微調)
VISIBILITY_THRESHOLD = 0.5

TARGET_COUNT = 5

fontpath = "NotoSansTC-Regular.ttf"
font = ImageFont.truetype(fontpath, 30)


def classify_state(hip, shoulder, wrist, wrists_together):
    angle = calculate_angle(hip, shoulder, wrist)

    # S0 獨立判斷，不受雙手是否握合影響
    # (坐姿準備動作時，雙手不一定會刻意握在一起)
    if angle <= S0_ANGLE_MAX:
        return "S0", angle

    if not wrists_together:
        return None, angle  # 雙手沒有合握，不視為前舉的有效動作(S1/S2)
    elif S1_ANGLE_MIN <= angle <= S1_ANGLE_MAX:
        return "S1", angle
    elif angle >= S2_ANGLE_MIN:
        return "S2", angle
    return None, angle


def angle_to_confidence(angle, target_angle, tolerance=30):
    diff = abs(angle - target_angle)
    return max(0.0, 100.0 * (1 - diff / tolerance))


def main():
    parser = argparse.ArgumentParser(description="手臂前舉(肩關節前屈)偵測程式")
    parser.add_argument(
        "--side", choices=["left", "right"], default="right",
        help="選擇要偵測的慣用手，預設為右手(right)",
    )
    args = parser.parse_args()
    landmark_names, _ = build_landmark_names(args.side)
    print(f"目前偵測手臂：{'左手' if args.side == 'left' else '右手'}")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 800, 600)

    counter = RepCounter(target_count=TARGET_COUNT)
    action_mapping = {"S0": "無動作", "S1": "雙臂前舉", "S2": "雙臂上舉", None: "動作中"}

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

            predicted_label = "S0"
            angle_value = 0.0
            confidence = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                hip, hip_vis = get_point(lm, landmark_names["HIP"])
                shoulder, sh_vis = get_point(lm, landmark_names["SHOULDER"])
                wrist, wr_vis = get_point(lm, landmark_names["WRIST"])
                left_wrist, lw_vis = get_point(lm, "LEFT_WRIST")
                right_wrist, rw_vis = get_point(lm, "RIGHT_WRIST")

                if min(hip_vis, sh_vis, wr_vis, lw_vis, rw_vis) >= VISIBILITY_THRESHOLD:
                    wrists_dist = euclidean_distance(left_wrist[:2], right_wrist[:2])
                    wrists_together = wrists_dist <= WRISTS_TOGETHER_MAX_DIST

                    state, angle_value = classify_state(hip, shoulder, wrist, wrists_together)
                    predicted_label = state  # None(過渡角度)保持None,不誤判成S0(中立姿勢)

                    if predicted_label == "S1":
                        confidence = angle_to_confidence(angle_value, 80, tolerance=30)
                    elif predicted_label == "S2":
                        confidence = angle_to_confidence(angle_value, 170, tolerance=30)

                draw_body_pose(image, results.pose_landmarks)

            is_s1 = predicted_label == "S1"
            is_s2 = predicted_label == "S2"
            is_neutral = predicted_label == "S0"
            counter.update(is_s1, is_s2, is_neutral, confidence_this_frame=confidence, timestamp=now)

            image = cv2.flip(image, 1)
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            predicted_label_chinese = action_mapping.get(predicted_label, "動作中")
            draw.text((10, 10), f"目前動作：{predicted_label_chinese}", fill=(0, 0, 0), font=font)

            prompt_text = {
                "NEUTRAL_WAIT": "請回到準備動作（立正站好，手臂自然垂下）",
                "AWAITING_S1": f"請做「{action_mapping['S1']}」動作",
                "AWAITING_S2": f"請做「{action_mapping['S2']}」動作",
            }.get(counter.phase, "")
            draw.text((10, 46), f"提示：{prompt_text}", fill=(0, 100, 200), font=font)

            draw.text((10, 82), f"角度：{angle_value:.1f}°", fill=(0, 0, 0), font=font)
            draw.text((10, 118), f"次數: {counter.total}", fill=(0, 0, 0), font=font)

            circle_color_s1 = (0, 255, 0) if counter.s1_done else (0, 0, 255)
            circle_color_s2 = (0, 255, 0) if counter.s2_done else (0, 0, 255)
            draw.text((450, 10), "前舉完成：", fill=(0, 0, 0), font=font)
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