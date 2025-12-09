# delivery_detection.py

import cv2
import numpy as np
import pyrealsense2 as rs
import signal, time
from ultralytics import YOLO
import math
import datetime
import json

import torch
from torchvision import models, transforms
import cv2, numpy as np

# 추가
# from groundingdino.util.inference import load_model, predict
import torchvision.transforms as T
import patient_info as info

# FPS 측정용
from collections import deque
import time
fps_history = deque(maxlen=10)


# 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# 모델 불러오기
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("gown_classifier.pth", map_location="cpu"))
model.eval()


# ====== 설정 ======
WIN = "Patient-Gown + Marker (ESC/q)"
YOLO_WEIGHTS = "yolov8n.pt"
YOLO_DOOR_WEIGHTS = "door_yolov8n.pt"
PERSON_CONF = 0.5
DOOR_CONF = 0.85
COLOR_RES = (640, 480)
FPS = 60
USE_DEPTH = True
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# ==================

stop = False
last_mid = None


def _sigint(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, _sigint)


# ---------- 환자복 판별 ----------
def is_patient_gown(crop_bgr: np.ndarray) -> bool:
    if crop_bgr.size == 0:
        return False

    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(out, 1).item()

    return pred == 0  # 0=gown, 1=normal


# ---------- QR/ArUco ----------
try:
    from pyzbar import pyzbar
    HAVE_PYZBAR = True
except Exception:
    HAVE_PYZBAR = False

qrd = cv2.QRCodeDetector()


def decode_markers(bgr):
    out = {'qr': [], 'aruco': []}

    try:
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(bgr)
        if ids is not None:
            for i, cid in enumerate(ids.flatten()):
                cs = corners[i].astype(int).reshape(-1, 2)
                out['aruco'].append((int(cid), cs))
    except Exception:
        pass

    return out


def clamp_box(x1,y1,x2,y2, W,H):
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)


def draw_poly(img, pts, color=(0,255,0), thickness=2):
    if pts is not None and len(pts) >= 4:
        cv2.polylines(img, [pts], True, color, thickness)


# ===============================================
#                    main()
# ===============================================
def main():
    global stop

    # YOLO 로드
    person_model = YOLO(YOLO_WEIGHTS)
    door_model = YOLO(YOLO_DOOR_WEIGHTS)
    _ = person_model.predict(np.zeros((480,640,3), dtype=np.uint8), verbose=False)
    _ = door_model.predict(np.zeros((480,640,3), dtype=np.uint8), verbose=False)

    # RealSense 파이프라인
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, FPS)

    if USE_DEPTH:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
        align = rs.align(rs.stream.color)

    profile = pipe.start(cfg)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    post_mid = 0

    try:
        while not stop:
            start_time = time.time()

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            frames = pipe.wait_for_frames()
            if USE_DEPTH:
                frames = align.process(frames)

            color_f = frames.get_color_frame()
            if not color_f:
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
                continue

            img = np.asanyarray(color_f.get_data())
            H, W = img.shape[:2]
            depth_f = frames.get_depth_frame() if USE_DEPTH else None

            # (1) 사람 탐지
            res = person_model.predict(img, conf=PERSON_CONF, verbose=False)

            for r in res:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2, W,H)

                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

                    # 상체 ROI
                    ph = y2 - y1
                    torso_y2 = y1 + int(ph * 0.65)
                    tx1, ty1, tx2, ty2 = x1, y1, x2, max(y1+1, torso_y2)
                    torso = img[ty1:ty2, tx1:tx2]

                    gown = is_patient_gown(torso)
                    label = f"person ({'gown' if gown else 'clothes'})"
                    cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (0,255,0) if gown else (0,165,255), 2)

                    if not gown:
                        continue

                    # 마커 탐지
                    roi = img[y1:y2, x1:x2]
                    scale_up = 1.0
                    if max(roi.shape[:2]) < 420:
                        roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
                        scale_up = 1.5

                    det = decode_markers(roi)
                    aruco_list = det['aruco']
                    if len(aruco_list) == 0:
                        continue

                    for mid, corners in aruco_list:
                        corners = corners.reshape(-1,2)
                        pts = (corners / scale_up).astype(int) + np.array([x1,y1])
                        draw_poly(img, pts, (0,255,255), 2)

                        cxy = pts.mean(axis=0).astype(int)
                        dist_str = ""

                        if USE_DEPTH and depth_f:
                            d = depth_f.get_distance(int(cxy[0]), int(cxy[1]))
                            if d > 0:
                                dist_str = f" | {d:.2f}m"

                        cv2.putText(img, f"ArUco : {mid}{dist_str}",
                                    (pts[0,0]+30, pts[0,1]-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                        
                        cv2.circle(img, tuple(cxy), 4, (0,255,255), -1)

                        patient_data = info.get_patient_info(str(mid))
                        myname = "Unknown" if (not patient_data or not isinstance(patient_data, dict)) \
                                  else patient_data.get("final_name", "Unknown")

                        cv2.putText(img, f"Name : {myname}",
                                    (pts[0,0]+30, pts[0,1]-26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

                    mid = aruco_list[0][0]

                    if post_mid != mid:
                        crop = img[y1:y2, x1:x2]

                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_name = f"patient_{mid}_{ts}"
                        jpg_path = f"detections/{base_name}.jpg"
                        json_path = f"detections/{base_name}.json"

                        cv2.imwrite(jpg_path, crop)

                        bbox_data = {
                            f"patient_{mid}": {
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "phrase": "",
                                "caption": ""
                            }
                        }

                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(bbox_data, f, ensure_ascii=False, indent=4)

                        print(f"New patient detected (ID={mid}) → Saved {jpg_path}")
                        post_mid = mid


            # (2) 문 탐지
            res_door = door_model.predict(img, conf=DOOR_CONF, verbose=False)
            for r in res_door:
                for b in r.boxes:
                    x1_d, y1_d, x2_d, y2_d = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(img, (x1_d,y1_d), (x2_d,y2_d), (0,255,255), 2)
                    cv2.putText(img, "Door", (x1_d,y1_d-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                    crop_d = img[y1_d:y2_d, x1_d:x2_d]

                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"door_{ts}"
                    jpg_path = f"detections/{base_name}.jpg"
                    json_path = f"detections/{base_name}.json"

                    cv2.imwrite(jpg_path, crop_d)

                    bbox_data = {
                        "door": {
                            "bbox": [float(x1_d), float(y1_d), float(x2_d), float(y2_d)],
                            "phrase": "",
                            "caption": ""
                        }
                    }

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(bbox_data, f, ensure_ascii=False, indent=4)

                    print(f"New door detected → Saved {jpg_path}")


            # FPS 표시
            end_time = time.time()
            frame_time = end_time - start_time

            if frame_time > 0:
                fps_history.append(1.0 / frame_time)
                avg_fps = sum(fps_history) / len(fps_history)
            else:
                avg_fps = 0.0

            cv2.putText(img, f"FPS: {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow(WIN, img)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
