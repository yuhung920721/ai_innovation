# shoulder_horizontal_adduction.py
# ------------------------------------------------------------
# 動作名稱：手臂平舉（肩關節水平內收 Shoulder Horizontal Adduction）
# 原程式檔名：Flat.py
# 復健功用：改善肩關節後側關節囊活動度、緩解僵硬及疼痛
#
# 【關節點】
#   1) RIGHT_SHOULDER - RIGHT_ELBOW - RIGHT_WRIST → 手肘彎曲角度
#   2) RIGHT_HIP - RIGHT_SHOULDER - RIGHT_WRIST   → 手臂離開軀幹的抬舉角度(確認手臂在肩高)
#   3) RIGHT_WRIST 與 對側肩膀(OPPOSITE_SHOULDER) 的距離 → 確認手掌是否真的靠近對側肩膀
#
# 【判斷邏輯】
#   S2「手臂平舉」（起始姿勢，手肘打直、手臂側平舉與肩同高）：
#       elbow_angle ≥ 150°（手肘接近打直）
#       lift_angle 落在 70°~110°（手臂離開軀幹約90度，即與肩同高側平舉）
#   S1「手掌碰肩」（動作終點，手肘彎曲、手掌跨過身體靠近對側肩膀）：
#       elbow_angle ≤ 100°（手肘明顯彎曲）
#       wrist 與對側肩膀的距離 < 門檻（手掌真的靠近對側肩膀，而非只是彎曲手肘）
# ------------------------------------------------------------

import argparse
import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ---- AI 復健報告服務（提前載入，避免完成動作那一瞬間才第一次載入套件造成卡頓）----
try:
    from services.report_service import submit_session, warm_up
    _report_service_ready = True
except ImportError as e:
    print(f"[AI報告] 找不到 services 模組，AI報告功能停用: {e}")
    _report_service_ready = False

    def submit_session(*args, **kwargs):
        pass

    def warm_up():
        pass


from pose_utils import (
    mp_pose,
    get_point,
    calculate_angle,
    euclidean_distance,
    RepCounter,
    build_landmark_names,
    draw_body_pose,
)

LIFT_ANGLE_S2_MIN, LIFT_ANGLE_S2_MAX = 70, 110   # S2: 手臂與肩同高(約90度抬舉)
WRIST_NEAR_OPPOSITE_SHOULDER_DIST = 0.18          # S1: 手腕靠近對側肩膀的距離門檻
WRIST_FAR_FROM_OPPOSITE_SHOULDER_DIST = 0.35      # S2: 手腕須遠離對側肩膀(確認真的伸展到側邊)
S0_LIFT_ANGLE_MAX = 45      # S0: 手臂垂下(準備動作)的抬舉角度門檻
                             # 注意：此處lift_angle用「髖-肩-腕」計算(非髖-肩-肘)，
                             # 前臂自然垂放時會有一些偏移，門檻本來就會比只看上臂的
                             # 側舉/前舉(15度)高一些，是正常現象，非誤差
VISIBILITY_THRESHOLD = 0.5

TARGET_COUNT = 5

fontpath = "NotoSansTC-Regular.ttf"
font = ImageFont.truetype(fontpath, 30)


def classify_state(hip, shoulder, elbow, wrist, opposite_shoulder):
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    lift_angle = calculate_angle(hip, shoulder, wrist)
    wrist_to_opp_shoulder = euclidean_distance(wrist[:2], opposite_shoulder[:2])

    # S0: 手臂垂下的準備動作
    if lift_angle <= S0_LIFT_ANGLE_MAX:
        return "S0", elbow_angle, lift_angle

    # 注意：手肘角度在這個動作裡不當作硬性判斷門檻（實測發現手肘角度在
    # S1/S2 之間的關係因人而異，跟原本假設不一致），只用手腕與對側肩膀
    # 的距離、以及抬舉角度來判斷，手肘角度保留給計分(角度標準度)使用。
    if wrist_to_opp_shoulder <= WRIST_NEAR_OPPOSITE_SHOULDER_DIST:
        return "S1", elbow_angle, lift_angle
    elif (
        wrist_to_opp_shoulder >= WRIST_FAR_FROM_OPPOSITE_SHOULDER_DIST
        and LIFT_ANGLE_S2_MIN <= lift_angle <= LIFT_ANGLE_S2_MAX
    ):
        return "S2", elbow_angle, lift_angle
    return None, elbow_angle, lift_angle


def angle_to_confidence(value, target, tolerance=30):
    diff = abs(value - target)
    return max(0.0, 100.0 * (1 - diff / tolerance))


def main():
    parser = argparse.ArgumentParser(description="手臂平舉(肩關節水平內收)偵測程式")
    parser.add_argument(
        "--side", choices=["left", "right"], default="right",
        help="選擇要偵測的慣用手，預設為右手(right)",
    )
    args = parser.parse_args()
    landmark_names, opposite_shoulder_name = build_landmark_names(args.side)
    print(f"目前偵測手臂：{'左手' if args.side == 'left' else '右手'}")

    warm_up()  # 提前載入 AI 報告會用到的套件，避免完成動作那一刻才卡頓

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 800, 600)

    counter = RepCounter(target_count=TARGET_COUNT)
    action_mapping = {"S0": "無動作", "S1": "手掌碰肩", "S2": "手臂平舉", None: "動作中"}
    session_start_time = time.time()  # 用來計算本次訓練耗時
    report_submitted = False          # 確保完成後只觸發一次 AI 報告

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
            elbow_angle = 0.0
            lift_angle = 0.0
            confidence = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                hip, hip_vis = get_point(lm, landmark_names["HIP"])
                shoulder, sh_vis = get_point(lm, landmark_names["SHOULDER"])
                elbow, el_vis = get_point(lm, landmark_names["ELBOW"])
                wrist, wr_vis = get_point(lm, landmark_names["WRIST"])
                opp_shoulder, opp_vis = get_point(lm, opposite_shoulder_name)

                if min(hip_vis, sh_vis, el_vis, wr_vis, opp_vis) >= VISIBILITY_THRESHOLD:
                    state, elbow_angle, lift_angle = classify_state(
                        hip, shoulder, elbow, wrist, opp_shoulder
                    )
                    predicted_label = state  # None(過渡角度)保持None,不誤判成S0(中立姿勢)

                    if predicted_label == "S1":
                        # 實測S1(手掌碰肩)最標準時手肘角度約150度
                        confidence = angle_to_confidence(elbow_angle, 150, tolerance=30)
                    elif predicted_label == "S2":
                        # 實測S2(手臂平舉)最標準時手肘角度約110度
                        confidence = angle_to_confidence(elbow_angle, 110, tolerance=30)

                draw_body_pose(image, results.pose_landmarks)

            # 正確動作順序：手掌碰肩(彎曲,S1) -> 手臂平舉(伸直,S2)
            is_s1 = predicted_label == "S1"  # 第一步：手掌碰肩
            is_s2 = predicted_label == "S2"  # 第二步：手臂平舉
            is_neutral = predicted_label == "S0"
            counter.update(is_s1, is_s2, is_neutral, confidence_this_frame=confidence, timestamp=now)

            # ---- 完成瞬間觸發 AI 復健報告（只會執行這一次）----
            if counter.finished and not report_submitted:
                report_submitted = True
                # submit_session 內部是 fire-and-forget（含資料庫寫入都在背景執行緒），
                # 這裡呼叫幾乎是零成本，不會造成畫面卡頓
                submit_session(
                    exercise_type="肩關節水平內收（手臂平舉）",
                    side=args.side,
                    target_count=counter.target_count,
                    completed_count=counter.total,
                    average_score=counter.average_score,
                    duration_sec=time.time() - session_start_time,
                    rep_history=counter.rep_history,
                )

            image = cv2.flip(image, 1)
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            predicted_label_chinese = action_mapping.get(predicted_label, "動作中")
            draw.text((10, 10), f"目前動作：{predicted_label_chinese}", fill=(0, 0, 0), font=font)

            prompt_text = {
                "NEUTRAL_WAIT": "請回到準備動作（立正站好，手臂自然垂下）",
                "AWAITING_S1": f"請做「{action_mapping['S1']}」動作",  # 第一步：手掌碰肩
                "AWAITING_S2": f"請做「{action_mapping['S2']}」動作",  # 第二步：手臂平舉
            }.get(counter.phase, "")
            draw.text((10, 46), f"提示：{prompt_text}", fill=(0, 100, 200), font=font)

            draw.text((10, 82), f"手肘角度：{elbow_angle:.1f}°　抬舉角度：{lift_angle:.1f}°", fill=(0, 0, 0), font=font)
            draw.text((10, 118), f"次數: {counter.total}", fill=(0, 0, 0), font=font)

            circle_color_s1 = (0, 255, 0) if counter.s1_done else (0, 0, 255)
            circle_color_s2 = (0, 255, 0) if counter.s2_done else (0, 0, 255)
            draw.text((450, 10), "碰肩完成：", fill=(0, 0, 0), font=font)
            draw.ellipse([(600, 15), (620, 35)], fill=circle_color_s1)
            draw.text((450, 46), "平舉完成：", fill=(0, 0, 0), font=font)
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