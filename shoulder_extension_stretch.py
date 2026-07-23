# shoulder_extension_stretch.py
# ------------------------------------------------------------
# 動作名稱：手臂後舉（肩後伸暨觸頸伸展 / 肩關節外旋活動度訓練）
# 原程式檔名：Back.py
# 動作說明：雙手握拳伸直至頭頂，再將雙手放下至後頸
# 復健功用：預防肌肉萎縮及拉伸上臂前後側肌肉、肩關節外旋活動度
#
# 【關節點】
#   1) RIGHT_HIP - RIGHT_SHOULDER - RIGHT_WRIST   → 手臂上舉角度
#   2) RIGHT_SHOULDER - RIGHT_ELBOW - RIGHT_WRIST → 手肘彎曲角度
#   3) RIGHT_WRIST 與 RIGHT_EAR 的距離             → 手是否貼近後頸/耳側
#
# 【判斷邏輯】
#   S1「握拳向前」（雙手伸直至頭頂）：
#       shoulder_angle ≥ 150°（手臂幾乎打直舉過頭頂）
#       elbow_angle ≥ 150°（手肘打直）
#   S2「握拳向後」（雙手放下至後頸）：
#       elbow_angle 落在 30°~90°（手肘明顯彎曲）
#       wrist 與同側耳朵的距離 < 門檻（手已收回貼近後頸/耳側）
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

SHOULDER_ANGLE_S1_MIN = 150     # S1: 手臂舉過頭頂的角度門檻
ELBOW_ANGLE_S1_MIN = 100        # S1: 手肘打直門檻
ELBOW_ANGLE_S2_MIN, ELBOW_ANGLE_S2_MAX = 15, 100   # S2: 手肘彎曲區間（貼頸時彎曲幅度較大）
WRIST_NEAR_EAR_DIST = 0.20      # S2: 手腕貼近耳側/後頸的距離門檻
S0_SHOULDER_ANGLE_MAX = 50      # S0: 準備動作(坐姿)的抬舉角度門檻，與手臂前舉設計一致
VISIBILITY_THRESHOLD = 0.3      # 降低可信度門檻(原0.5)，手貼後頸時手腕/耳朵容易被遮擋，
                                 # 可信度會偏低，門檻太嚴格會導致角度無法計算(顯示變成0)

TARGET_COUNT = 5

fontpath = "NotoSansTC-Regular.ttf"
font = ImageFont.truetype(fontpath, 30)


def classify_state(hip, shoulder, elbow, wrist, ear):
    shoulder_angle = calculate_angle(hip, shoulder, wrist)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    wrist_to_ear = euclidean_distance(wrist[:2], ear[:2])

    # S0: 準備動作(手臂垂放/坐姿手放大腿)，先前完全沒有定義
    if shoulder_angle <= S0_SHOULDER_ANGLE_MAX:
        return "S0", shoulder_angle, elbow_angle

    if shoulder_angle >= SHOULDER_ANGLE_S1_MIN and elbow_angle >= ELBOW_ANGLE_S1_MIN:
        return "S1", shoulder_angle, elbow_angle
    elif (
        ELBOW_ANGLE_S2_MIN <= elbow_angle <= ELBOW_ANGLE_S2_MAX
        and wrist_to_ear <= WRIST_NEAR_EAR_DIST
    ):
        return "S2", shoulder_angle, elbow_angle
    return None, shoulder_angle, elbow_angle


def angle_to_confidence(value, target, tolerance=30):
    diff = abs(value - target)
    return max(0.0, 100.0 * (1 - diff / tolerance))


def main():
    parser = argparse.ArgumentParser(description="手臂後舉(肩後伸暨觸頸伸展)偵測程式")
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
    action_mapping = {"S0": "無動作", "S1": "雙臂上舉", "S2": "雙臂後舉", None: "動作中"}

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
            shoulder_angle = 0.0
            elbow_angle = 0.0
            confidence = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                hip, hip_vis = get_point(lm, landmark_names["HIP"])
                shoulder, sh_vis = get_point(lm, landmark_names["SHOULDER"])
                elbow, el_vis = get_point(lm, landmark_names["ELBOW"])
                wrist, wr_vis = get_point(lm, landmark_names["WRIST"])
                ear, ear_vis = get_point(lm, landmark_names["EAR"])

                if min(hip_vis, sh_vis, el_vis, wr_vis, ear_vis) >= VISIBILITY_THRESHOLD:
                    state, shoulder_angle, elbow_angle = classify_state(
                        hip, shoulder, elbow, wrist, ear
                    )
                    predicted_label = state  # None(過渡角度)保持None,不誤判成S0(中立姿勢)

                    if predicted_label == "S1":
                        # 依實測數據，肩角150~170度、肘角105~125度才是真正做到
                        # 最標準的姿勢(人體結構上握拳過頭很難打到理論值180度全直)
                        # 兩個指標都納入計分，取平均
                        shoulder_conf = angle_to_confidence(shoulder_angle, 170, tolerance=30)
                        elbow_conf = angle_to_confidence(elbow_angle, 110, tolerance=30)
                        confidence = (shoulder_conf + elbow_conf) / 2
                    elif predicted_label == "S2":
                        # 依實測數據，肩角約100度、肘角約80度才是真正做到最標準的貼頸姿勢
                        shoulder_conf_s2 = angle_to_confidence(shoulder_angle, 100, tolerance=30)
                        elbow_conf_s2 = angle_to_confidence(elbow_angle, 80, tolerance=30)
                        confidence = (shoulder_conf_s2 + elbow_conf_s2) / 2

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
            draw.text((10, 82), "※ 本動作請側身面對鏡頭，避免手部被頭部遮擋", fill=(200, 0, 0), font=font)

            draw.text((10, 118), f"肩角:{shoulder_angle:.0f}° 肘角:{elbow_angle:.0f}°", fill=(0, 0, 0), font=font)
            draw.text((10, 154), f"次數: {counter.total}", fill=(0, 0, 0), font=font)

            circle_color_s1 = (0, 255, 0) if counter.s1_done else (0, 0, 255)
            circle_color_s2 = (0, 255, 0) if counter.s2_done else (0, 0, 255)
            draw.text((450, 10), "上舉完成：", fill=(0, 0, 0), font=font)
            draw.ellipse([(600, 15), (620, 35)], fill=circle_color_s1)
            draw.text((450, 46), "後舉完成：", fill=(0, 0, 0), font=font)
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