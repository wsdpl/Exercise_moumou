import email
import email.parser
import email.message
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QComboBox, QVBoxLayout, QHBoxLayout, QProgressBar,
    QTextEdit, QGroupBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image, ImageDraw, ImageFont

# ================== 模型加载 ==================
import os
import sys

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else sys.argv[0]))
    return os.path.join(base_path, relative_path)


MODEL_PATH = get_resource_path("best.pt")
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
model = YOLO(MODEL_PATH)

# ================== 关键点工具函数 ==================
def get_best_kpt(kpts, confs, left_idx, right_idx, conf_thresh=0.5):
    left_conf = float(confs[left_idx]) if confs is not None and left_idx < len(confs) else 0.0
    right_conf = float(confs[right_idx]) if confs is not None and right_idx < len(confs) else 0.0
    if left_conf >= conf_thresh and left_conf >= right_conf:
        return kpts[left_idx], left_conf
    elif right_conf >= conf_thresh:
        return kpts[right_idx], right_conf
    elif left_conf > 0 or right_conf > 0:
        return (kpts[left_idx], left_conf) if left_conf >= right_conf else (kpts[right_idx], right_conf)
    return kpts[left_idx], 0.0

def is_kpt_valid(pt, conf=None, conf_thresh=0.3):
    if conf is not None and float(conf) < conf_thresh:
        return False
    if np.any(np.isnan(pt)):
        return False
    if abs(float(pt[0])) < 1e-6 and abs(float(pt[1])) < 1e-6:
        return False
    return True

def validate_keypoints(required_pairs, kpts, confs):
    for name, (left_idx, right_idx) in required_pairs:
        pt, conf = get_best_kpt(kpts, confs, left_idx, right_idx)
        if not is_kpt_valid(pt, conf):
            return False, "关键点检测不可靠，请调整姿势或光线"
    return True, ""

def cv2_put_text(img, text, pos, color=(0, 0, 0), fontSize=24):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(FONT_PATH, fontSize)
    except:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        color = (0, 0, 0)
    draw.text(pos, text, color, font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# ================== 关键点绘制工具 ==================
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def draw_pose(img, kpts, confs=None, conf_thresh=0.3):
    overlay = img.copy()
    for (i, j) in SKELETON_CONNECTIONS:
        if i >= len(kpts) or j >= len(kpts): continue
        if not is_kpt_valid(kpts[i], confs[i] if confs is not None else None, conf_thresh): continue
        if not is_kpt_valid(kpts[j], confs[j] if confs is not None else None, conf_thresh): continue
        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
        pt2 = (int(kpts[j][0]), int(kpts[j][1]))
        cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
    for i in range(min(len(kpts), 17)):
        conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        if conf >= conf_thresh:
            x, y = int(kpts[i][0]), int(kpts[i][1])
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1)
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

def evaluate_squat(kpts, confs=None):
    """深蹲评分
    逻辑：只有真正在下蹲状态（膝角<130°）才评分，否则返回None不计入统计。
    评分标准：满分100，根据常见错误逐项扣分。"""
    valid, err = validate_keypoints([
        ("hip", (11, 12)), ("knee", (13, 14)), ("ankle", (15, 16)),
        ("shoulder", (5, 6))
    ], kpts, confs)
    if not valid:
        return 0, [err]

    hip, _ = get_best_kpt(kpts, confs, 11, 12)
    knee, _ = get_best_kpt(kpts, confs, 13, 14)
    ankle, _ = get_best_kpt(kpts, confs, 15, 16)
    shoulder, _ = get_best_kpt(kpts, confs, 5, 6)

    knee_angle = calculate_angle(hip, knee, ankle)
    torso_angle = calculate_angle(shoulder, hip, knee)

    if knee_angle >= 150:
        return None, ["站立姿态"]
    if knee_angle >= 130:
        return None, ["微蹲姿态"]

    score = 100
    feedback = []

    # 1. 下蹲深度
    if knee_angle > 120:
        feedback.append(f"蹲得太浅（膝关节角度{knee_angle:.0f}°，建议≤90°）")
        score -= 40
    elif knee_angle > 100:
        feedback.append(f"下蹲深度不足（膝关节角度{knee_angle:.0f}°，可再深一些）")
        score -= 20
    elif knee_angle > 90:
        feedback.append(f"接近标准深度（膝关节角度{knee_angle:.0f}°，可再低一点）")
        score -= 10

    # 2. 躯干稳定性
    if torso_angle < 45:
        feedback.append(f"严重弯腰驼背（躯干角{torso_angle:.0f}°，请保持挺直）")
        score -= 30
    elif torso_angle < 60:
        feedback.append(f"弯腰驼背（躯干角{torso_angle:.0f}°，注意挺胸收腹）")
        score -= 20
    elif torso_angle < 70:
        feedback.append(f"躯干略有前倾（躯干角{torso_angle:.0f}°，注意保持直立）")
        score -= 10

    # 3. 膝盖对齐（侧面视角下膝盖是否过度前移/内扣）
    leg_length = np.linalg.norm(np.array(hip, dtype=np.float32) - np.array(ankle, dtype=np.float32))
    if leg_length > 1:
        knee_forward = abs(knee[0] - hip[0]) / leg_length
        if knee_forward > 0.5:
            feedback.append("膝盖过度前移/内扣，注意膝盖方向与脚尖一致")
            score -= 15
        elif knee_forward > 0.35:
            feedback.append("膝盖略有偏移，注意控制方向")
            score -= 5

    if score >= 85:
        feedback.append("动作标准")
    return max(score, 0), feedback


def evaluate_pushup(kpts, confs=None):
    """俯卧撑评分
    逻辑：只有真正在下压状态（肘角<150°且身体成直线）才评分。"""
    valid, err = validate_keypoints([
        ("shoulder", (5, 6)), ("elbow", (7, 8)), ("wrist", (9, 10)),
        ("hip", (11, 12)), ("ankle", (15, 16))
    ], kpts, confs)
    if not valid:
        return 0, [err]

    shoulder, _ = get_best_kpt(kpts, confs, 5, 6)
    elbow, _ = get_best_kpt(kpts, confs, 7, 8)
    wrist, _ = get_best_kpt(kpts, confs, 9, 10)
    hip, _ = get_best_kpt(kpts, confs, 11, 12)
    ankle, _ = get_best_kpt(kpts, confs, 15, 16)

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    body_angle = calculate_angle(shoulder, hip, ankle)

    if elbow_angle >= 160:
        return None, ["休息姿态"]
    if body_angle < 140:
        return None, ["身体未撑起"]

    score = 100
    feedback = []

    # 1. 手臂弯曲度
    if elbow_angle > 130:
        feedback.append(f"手臂未弯到底（肘角{elbow_angle:.0f}°，建议降至90°以下）")
        score -= 40
    elif elbow_angle > 110:
        feedback.append(f"手臂弯曲不足（肘角{elbow_angle:.0f}°，可再低一些）")
        score -= 20
    elif elbow_angle > 90:
        feedback.append(f"接近标准深度（肘角{elbow_angle:.0f}°，可再低一点）")
        score -= 10

    # 2. 身体直线
    if body_angle < 150:
        feedback.append(f"严重塌腰/撅臀（身体角度{body_angle:.0f}°，请保持直线）")
        score -= 30
    elif body_angle < 165:
        feedback.append(f"塌腰/撅臀（身体角度{body_angle:.0f}°，注意收紧核心）")
        score -= 20
    elif body_angle < 170:
        feedback.append(f"身体略有弯曲（身体角度{body_angle:.0f}°，注意保持平直）")
        score -= 10

    # 3. 核心稳定性
    if 5 < len(kpts) and 6 < len(kpts):
        shoulder_diff = abs(float(kpts[5][1]) - float(kpts[6][1]))
    else:
        shoulder_diff = 0
    if 11 < len(kpts) and 12 < len(kpts):
        hip_diff = abs(float(kpts[11][1]) - float(kpts[12][1]))
    else:
        hip_diff = 0
    torso_len = np.linalg.norm(np.array(shoulder, dtype=np.float32) - np.array(hip, dtype=np.float32))
    if torso_len > 1:
        wobble_ratio = (shoulder_diff + hip_diff) / (2 * torso_len)
        if wobble_ratio > 0.15:
            feedback.append("躯干晃动/侧弯明显，注意保持身体稳定")
            score -= 15
        elif wobble_ratio > 0.08:
            feedback.append("躯干有轻微晃动，注意控制稳定性")
            score -= 5

    if score >= 85:
        feedback.append("动作标准")
    return max(score, 0), feedback


def evaluate_situp(kpts, confs=None):
    """仰卧起坐评分
    逻辑：只有真正在起身状态（rise_ratio>0.15）才评分。"""
    valid, err = validate_keypoints([
        ("shoulder", (5, 6)), ("hip", (11, 12)), ("knee", (13, 14))
    ], kpts, confs)
    if not valid:
        return 0, [err]

    shoulder, _ = get_best_kpt(kpts, confs, 5, 6)
    hip, _ = get_best_kpt(kpts, confs, 11, 12)
    knee, _ = get_best_kpt(kpts, confs, 13, 14)

    hip_angle = calculate_angle(shoulder, hip, knee)
    shoulder_rise = hip[1] - shoulder[1]
    torso_length = np.linalg.norm(np.array(shoulder, dtype=np.float32) - np.array(hip, dtype=np.float32))
    rise_ratio = shoulder_rise / torso_length if torso_length > 1 else 0.0

    if rise_ratio < 0.05:
        return None, ["躺平姿态"]
    if rise_ratio < 0.15:
        return None, ["微起姿态"]

    score = 100
    feedback = []

    # 1. 起身幅度
    if rise_ratio < 0.25:
        feedback.append(f"起身幅度严重不足（上升比{rise_ratio:.2f}，需充分坐起）")
        score -= 40
    elif rise_ratio < 0.35:
        feedback.append(f"起身幅度不足（上升比{rise_ratio:.2f}，需再高一些）")
        score -= 25
    elif rise_ratio < 0.45:
        feedback.append(f"起身幅度一般（上升比{rise_ratio:.2f}，可再高一些）")
        score -= 15

    # 2. 髋关节弯曲
    if hip_angle > 120:
        feedback.append(f"髋关节未充分弯曲（髋角{hip_angle:.0f}°，建议腹部发力卷起）")
        score -= 25
    elif hip_angle > 100:
        feedback.append(f"髋关节弯曲不足（髋角{hip_angle:.0f}°，注意腹部发力）")
        score -= 15
    elif hip_angle > 85:
        feedback.append(f"接近标准（髋角{hip_angle:.0f}°，可再弯曲一些）")
        score -= 5

    # 3. 躯干稳定性
    if 5 < len(kpts) and 6 < len(kpts):
        shoulder_diff = abs(float(kpts[5][1]) - float(kpts[6][1]))
    else:
        shoulder_diff = 0
    if torso_length > 1:
        wobble_ratio = shoulder_diff / torso_length
        if wobble_ratio > 0.2:
            feedback.append("躯干歪斜明显，注意保持上身正直")
            score -= 15
        elif wobble_ratio > 0.1:
            feedback.append("躯干略有歪斜，注意保持对称发力")
            score -= 5

    if score >= 85:
        feedback.append("动作标准")
    return max(score, 0), feedback

def get_training_advice(feedback):
    advice_parts = []
    for fb in feedback:
        # 深蹲
        if "蹲得太浅" in fb:
            advice_parts.append("训练建议：下蹲至髋部低于膝盖，膝关节角度控制在90°左右，保持腰背挺直缓慢下蹲。")
        elif "下蹲深度不足" in fb:
            advice_parts.append("训练建议：继续下蹲，目标是膝关节角度≤90°，感受大腿和臀部发力。")
        elif "接近标准深度" in fb:
            advice_parts.append("训练建议：已经接近标准深度，再稍微下蹲一点即可达标，保持当前节奏。")
        elif "严重弯腰驼背" in fb:
            advice_parts.append("训练建议：下蹲时收紧核心肌群，保持上半身挺直，目光平视前方，避免含胸驼背。")
        elif "弯腰驼背" in fb:
            advice_parts.append("训练建议：注意挺胸收腹，上半身尽量保持直立，可适当将重心放在脚后跟。")
        elif "躯干略有前倾" in fb:
            advice_parts.append("训练建议：轻微调整姿势，注意保持躯干正直，核心收紧。")
        elif "膝盖过度前移" in fb or "膝盖内扣" in fb:
            advice_parts.append("训练建议：下蹲时膝盖方向应与脚尖一致，避免膝盖内扣或过度前移，双脚站稳。")
        elif "膝盖略有偏移" in fb:
            advice_parts.append("训练建议：注意控制膝盖方向顺着脚尖，避免内扣，加强臀中肌训练有助于改善。")
        # 俯卧撑
        elif "手臂未弯到底" in fb:
            advice_parts.append("训练建议：俯卧撑下沉时手臂充分弯曲，胸部贴近地面，推起时手臂完全伸直。")
        elif "手臂弯曲不足" in fb:
            advice_parts.append("训练建议：继续下沉，目标是肘角≤90°，胸部尽量贴近地面再推起。")
        elif "接近标准深度" in fb and "肘角" in fb:
            advice_parts.append("训练建议：已经接近标准深度，再稍微下沉一点即可达标。")
        elif "严重塌腰" in fb:
            advice_parts.append("训练建议：俯卧撑全程保持从头到脚一条直线，收紧核心和臀部，避免塌腰或撅臀。")
        elif "塌腰" in fb or "撅臀" in fb:
            advice_parts.append("训练建议：注意收紧腹部和臀部，保持身体平直，避免腰部下塌。")
        elif "身体略有弯曲" in fb:
            advice_parts.append("训练建议：轻微调整姿势，注意保持身体成一条直线。")
        elif "躯干晃动" in fb or "侧弯" in fb:
            advice_parts.append("训练建议：注意保持身体稳定不晃动，核心肌群持续收紧，控制动作节奏。")
        # 仰卧起坐
        elif "起身幅度严重不足" in fb:
            advice_parts.append("训练建议：仰卧起坐需充分坐起至上背部完全离开地面，感受腹肌收缩发力。")
        elif "起身幅度不足" in fb:
            advice_parts.append("训练建议：继续起身，目标是肩胛骨完全离开地面，动作幅度要充分。")
        elif "起身幅度一般" in fb:
            advice_parts.append("训练建议：当前起身幅度尚可，尝试坐得更高，让肩胛骨完全离开地面。")
        elif "髋关节未充分弯曲" in fb:
            advice_parts.append("训练建议：仰卧起坐时腹部发力卷起，髋关节充分弯曲，避免借助腰部或惯性。")
        elif "髋关节弯曲不足" in fb:
            advice_parts.append("训练建议：注意腹部主动发力卷起躯干，减少髋关节代偿。")
        elif "接近标准" in fb and "髋角" in fb:
            advice_parts.append("训练建议：已经接近标准，注意保持腹部发力卷起的感觉。")
        elif "躯干歪斜" in fb:
            advice_parts.append("训练建议：仰卧起坐时保持上身正直，左右对称发力，避免歪斜。")
        elif "动作标准" in fb:
            advice_parts.append("训练建议：动作姿势规范，保持当前动作节奏继续练习即可。")
    return "\n\n".join(advice_parts) if advice_parts else "等待检测..."

class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_progress = pyqtSignal(int)
    update_advice = pyqtSignal(str)
    task_finished = pyqtSignal()

    def __init__(self, video_path, output_path, action):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.action = action
        self.running = True
        self.total_score = 0
        self.detected_count = 0
        self.feedback_counts = {}
        self.detect_interval = 3

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, real_fps, (w, h))

        count = 0
        last_kpts = None
        last_confs = None

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            # 更新进度
            progress = int((count / total_frames) * 100)
            self.update_progress.emit(progress)

            if count % self.detect_interval == 1:
                kpts, confs, score, fb = self._detect_frame(frame)
                if kpts is not None:
                    last_kpts = kpts
                    last_confs = confs
                    if score is not None:
                        self.total_score += score
                        self.detected_count += 1
                        for fb_item in fb:
                            self.feedback_counts[fb_item] = self.feedback_counts.get(fb_item, 0) + 1

            if last_kpts is not None:
                display = draw_pose(frame.copy(), last_kpts, last_confs)
            else:
                display = frame.copy()

            out.write(display)
            self.update_frame.emit(display)

        cap.release()
        out.release()

        # 输出综合评估结果
        if self.detected_count > 0:
            avg_score = self.total_score / self.detected_count
            sorted_fb = sorted(self.feedback_counts.items(), key=lambda x: -x[1])
            threshold = max(1, int(self.detected_count * 0.3))
            main_fb = [item for item, cnt in sorted_fb if cnt >= threshold and '✅' not in item]
            if not main_fb:
                main_fb = ['动作标准']
            advice_text = get_training_advice(main_fb)
            summary = f"=== 综合评估 (共检测{self.detected_count}帧) ===\n"
            summary += f"平均得分: {avg_score:.1f} 分\n\n"
            if main_fb:
                summary += "主要问题:\n"
                for item in main_fb:
                    c = dict(sorted_fb).get(item, 0)
                    pct = c / self.detected_count * 100
                    summary += f"  {item} ({pct:.0f}% 的帧)\n"
            summary += f"\n{advice_text}"
            self.update_advice.emit(summary)
        else:
            self.update_advice.emit("没有检测到有效体态，请检查视频是否正确")
        self.task_finished.emit()

    def _detect_frame(self, frame):
        """同步检测单帧，返回 (kpts, confs, score, feedback)"""
        try:
            h, w = frame.shape[:2]
            target = 480
            scale = target / max(w, h) if max(w, h) > target else 1.0
            if scale < 1:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                results = model.predict(small, conf=0.5, verbose=False, max_det=1, agnostic_nms=True)
            else:
                scale = 1.0
                results = model.predict(frame, conf=0.5, verbose=False, max_det=1, agnostic_nms=True)

            best_result = None
            max_area = 0
            for r in results:
                if len(r.boxes) == 0:
                    continue
                box = r.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_result = r

            if best_result is None:
                return None, None, None, None

            kpts = best_result.keypoints.xy[0].cpu().numpy()
            if scale != 1.0:
                kpts[:, 0] /= scale
                kpts[:, 1] /= scale
            confs = best_result.keypoints.conf[0].cpu().numpy() if best_result.keypoints.conf is not None else None

            valid_kpt_count = sum(
                1 for i in range(len(kpts))
                if not (np.any(np.isnan(kpts[i])) or (abs(kpts[i][0]) < 1 and abs(kpts[i][1]) < 1))
            )
            if valid_kpt_count < 5:
                return None, None, None, None

            if self.action == '深蹲':
                score, fb = evaluate_squat(kpts, confs)
            elif self.action == '俯卧撑':
                score, fb = evaluate_pushup(kpts, confs)
            else:
                score, fb = evaluate_situp(kpts, confs)

            return kpts, confs, score, fb
        except Exception as e:
            print(f'Detection error: {e}')
            return None, None, None, None

    def stop(self):
        self.running = False
class CameraThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_advice = pyqtSignal(str)
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.running = True
        self.total_score = 0
        self.detected_count = 0
        self.feedback_counts = {}
    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret: break
            results = model.predict(frame, conf=0.5, verbose=False, imgsz=640)
            best_result = None
            max_area = 0
            for r in results:
                if len(r.boxes) == 0: continue
                box = r.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_result = r
            if best_result is not None:
                kpts = best_result.keypoints.xy[0].cpu().numpy()
                confs = best_result.keypoints.conf[0].cpu().numpy() if best_result.keypoints.conf is not None else None
                valid_kpt_count = sum(1 for i in range(len(kpts)) if not (np.any(np.isnan(kpts[i])) or (abs(kpts[i][0]) < 1 and abs(kpts[i][1]) < 1)))
                if valid_kpt_count >= 5:
                    if self.action == "深蹲":
                        score, fb = evaluate_squat(kpts, confs)
                    elif self.action == "俯卧撑":
                        score, fb = evaluate_pushup(kpts, confs)
                    else:
                        score, fb = evaluate_situp(kpts, confs)
                    if score is not None:
                        self.total_score += score
                        self.detected_count += 1
                        for fb_item in fb:
                            self.feedback_counts[fb_item] = self.feedback_counts.get(fb_item, 0) + 1
                    frame = best_result.plot()
            self.update_frame.emit(frame)
        cap.release()
        if self.detected_count > 0:
            avg_score = self.total_score / self.detected_count
            sorted_fb = sorted(self.feedback_counts.items(), key=lambda x: -x[1])
            threshold = max(1, int(self.detected_count * 0.3))
            main_fb = [item for item, count in sorted_fb if count >= threshold and "动作标准" not in item]
            if not main_fb:
                main_fb = ["动作标准"]
            advice_text = get_training_advice(main_fb)
            summary = f"└── 综合评估 (共检测{self.detected_count}帧) ──┘\n"
            summary += f"平均得分: {avg_score:.1f} 分\n\n"
            if main_fb:
                summary += "主要问题:\n"
                for item in main_fb:
                    count = dict(sorted_fb).get(item, 0)
                    pct = count / self.detected_count * 100
                    summary += f"  {item} ({pct:.0f}% 的帧)\n"
            summary += f"\n{advice_text}"
            self.update_advice.emit(summary)
        else:
            self.update_advice.emit("未能检测到有效人体")
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("锻炼牛牛")
        self.setGeometry(80, 80, 1300, 800)
        self.setStyleSheet("background-color: #f5f7fa; color: black;")
        self.thread = None
        self.video_path = None
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(900, 550)
        self.video_label.setStyleSheet("QLabel { background-color: #ffffff; border-radius: 12px; color: black; font-size: 16px; }")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("请选择视频或开启摄像头")
        left_layout.addWidget(self.video_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { height: 14px; border-radius: 7px; background-color: #e0e0e0; } QProgressBar::chunk { background-color: #4CAF50; border-radius: 7px; }")
        left_layout.addWidget(self.progress_bar)
        act_group = QGroupBox("动作类型选择")
        act_group.setStyleSheet("color: black; font-size:14px;")
        act_layout = QVBoxLayout(act_group)
        self.action_combo = QComboBox()
        self.action_combo.addItems(["深蹲", "俯卧撑", "仰卧起坐"])
        self.action_combo.setStyleSheet("QComboBox { padding: 8px 12px; font-size: 14px; border-radius: 6px; background-color: white; color: black; }")
        act_layout.addWidget(self.action_combo)
        right_layout.addWidget(act_group)
        btn_group = QGroupBox("功能操作")
        btn_group.setStyleSheet("color: black;")
        btn_layout = QVBoxLayout(btn_group)
        self.select_btn = QPushButton("选择视频")
        self.start_btn = QPushButton("开始评估预览")
        self.camera_btn = QPushButton("开启摄像头")
        self.stop_btn = QPushButton("停止运行")
        btn_style = "QPushButton { padding: 10px; border-radius: 6px; font-size: 14px; color:black; background:#e9ecef; }"
        self.select_btn.setStyleSheet(btn_style)
        self.start_btn.setStyleSheet(btn_style)
        self.camera_btn.setStyleSheet(btn_style)
        self.stop_btn.setStyleSheet(btn_style)
        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.camera_btn)
        btn_layout.addWidget(self.stop_btn)
        right_layout.addWidget(btn_group)
        advice_group = QGroupBox("训练优化建议")
        advice_group.setStyleSheet("color: black;")
        advice_layout = QVBoxLayout(advice_group)
        self.advice_stack = QStackedWidget()
        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        load_path = get_resource_path("picture/load.jpg")
        load_pixmap = QPixmap(load_path)
        if load_pixmap.isNull():
            alt_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "picture", "load.jpg")
            load_pixmap = QPixmap(alt_path)
        if not load_pixmap.isNull():
            self.loading_label.setPixmap(load_pixmap.scaled(380, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.advice_stack.addWidget(self.loading_label)
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setFixedHeight(180)
        self.advice_text.setStyleSheet("QTextEdit{ background:#ffffff; border:1px solid #dee2e6; border-radius:8px; padding:8px; font-size:13px; color:black; }")
        self.advice_stack.addWidget(self.advice_text)
        advice_layout.addWidget(self.advice_stack)
        right_layout.addWidget(advice_group)
        right_layout.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.select_btn.clicked.connect(self.on_select_video)
        self.start_btn.clicked.connect(self.on_start)
        self.camera_btn.clicked.connect(self.on_open_camera)
        self.stop_btn.clicked.connect(self.on_stop)
    def on_select_video(self):
        self.on_stop()
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret:
                self.show_frame(frame)
            cap.release()
            self.advice_stack.setCurrentIndex(1)
            self.advice_text.clear()
    def on_start(self):
        if not self.video_path:
            return
        self.on_stop()
        self.show_loading_image()
        base, ext = os.path.splitext(self.video_path)
        out_path = f"{base}_result{ext}"
        action = self.action_combo.currentText()
        self.thread = VideoThread(self.video_path, out_path, action)
        self.thread.update_frame.connect(self.show_frame)
        self.thread.update_progress.connect(self.progress_bar.setValue)
        self.thread.update_advice.connect(self.show_advice)
        self.thread.task_finished.connect(self.on_finished)
        self.thread.start()
    def on_open_camera(self):
        self.on_stop()
        self.show_loading_image()
        action = self.action_combo.currentText()
        self.thread = CameraThread(action)
        self.thread.update_frame.connect(self.show_frame)
        self.thread.update_advice.connect(self.show_advice)
        self.thread.start()
    def on_stop(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        self.progress_bar.setValue(0)
    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    def show_loading_image(self):
        self.advice_stack.setCurrentIndex(0)

    def show_advice(self, text):
        self.advice_stack.setCurrentIndex(1)
        self.advice_text.setText(text)
        if "动作标准" in text:
            img_name = "good.png"
        else:
            img_name = "bad.jpg"
        img_path = get_resource_path("picture/" + img_name)
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            alt_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "picture", img_name)
            pixmap = QPixmap(alt_path)
        if not pixmap.isNull():
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def on_finished(self):
        pass
    def closeEvent(self, event):
        self.on_stop()
        event.ignore()
        self.hide()
        self.end_window = EndWindow(self)
        self.end_window.show()

class SplashWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("学生体能测试动作智能评估系统")
        self.setGeometry(80, 80, 1300, 800)
        self.setStyleSheet("background-color: #f5f7fa;")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splash_path = get_resource_path("picture/start.jpg")
        pixmap = QPixmap(splash_path)
        if pixmap.isNull():
            alt_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "picture", "start.jpg")
            pixmap = QPixmap(alt_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText("学生体能测试动作智能评估系统")
            self.image_label.setStyleSheet("font-size: 28px; color: #333; padding: 60px;")
        layout.addWidget(self.image_label)
        self.start_btn = QPushButton("和牛牛一起成为猛男")
        self.start_btn.setFixedSize(320, 60)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border-radius: 30px;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
        """)
        self.start_btn.clicked.connect(self.on_start)
        layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
    def on_start(self):
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

class EndWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("学生体能测试动作智能评估系统")
        self.setGeometry(80, 80, 1300, 800)
        self.setStyleSheet("background-color: #f5f7fa;")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        end_path = get_resource_path("picture/end.png")
        pixmap = QPixmap(end_path)
        if pixmap.isNull():
            alt_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "picture", "end.png")
            pixmap = QPixmap(alt_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText("学生体能测试动作智能评估系统")
            self.image_label.setStyleSheet("font-size: 28px; color: #333; padding: 60px;")
        layout.addWidget(self.image_label)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(40)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.leave_btn = QPushButton("狠心离开")
        self.leave_btn.setFixedSize(200, 60)
        self.leave_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.leave_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 30px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self.leave_btn.clicked.connect(self.on_leave)
        self.continue_btn = QPushButton("继续陪牛牛训练")
        self.continue_btn.setFixedSize(200, 60)
        self.continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 30px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self.continue_btn.clicked.connect(self.on_continue)
        btn_layout.addWidget(self.leave_btn)
        btn_layout.addWidget(self.continue_btn)
        layout.addLayout(btn_layout)
    def on_leave(self):
        QApplication.quit()
    def on_continue(self):
        self.close()
        self.main_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = SplashWindow()
    splash.show()
    sys.exit(app.exec())
