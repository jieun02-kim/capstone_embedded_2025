import os, time, threading, socket, json
from queue import Queue, Empty

import numpy as np
import cv2
from PIL import Image

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# =========================
# 설정
# =========================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LANG = "en"
MODE = "caption"
INFER_EVERY_SEC = 0.5
VIDEO_PATH = "./emg.mp4"  # ✅ 동영상 파일 경로
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

LLM_HOST, LLM_PORT = "127.0.0.1", 5555

# =========================
# 응급 키워드
# =========================
EMERGENCY_KEYWORDS = {
    "fallen": "fall", "fell": "fall", "falling": "fall", "collapse": "fall",
    "collapsed": "fall", "slipped": "fall",
    "fire": "fire", "flames": "fire", "burning": "fire", "explosion": "fire",
    "smoke": "smoke", "fumes": "smoke",
    "blood": "bleeding", "bleeding": "bleeding", "wound": "bleeding", "injury": "bleeding",
    "unconscious": "unconscious", "passed out": "unconscious", "fainted": "unconscious",
    "seizure": "seizure", "convulsion": "seizure",
    "choking": "choking", "not breathing": "choking", "difficulty breathing": "choking",
    "heart attack": "cardiac_arrest", "cardiac arrest": "cardiac_arrest", "stroke": "stroke",
    "scream": "scream", "cry for help": "scream",
}


def detect_emergency(text: str):
    for kw, emg_type in EMERGENCY_KEYWORDS.items():
        if kw.lower() in text.lower():
            return emg_type
    return None


def send_emergency(emg_type: str, confidence: float = 0.95):
    msg = {"type": emg_type, "confidence": confidence}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((LLM_HOST, LLM_PORT))
        sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        sock.close()
        print(f"[VLM] 🚨 Emergency sent to LLM: {msg}")
    except Exception as e:
        print(f"[VLM] Failed to send emergency: {e}")


# =========================
# 유틸
# =========================
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def put_multiline_text(img, text, org=(10, 30), line_height=20,
                       font_scale=0.5, color=(0, 255, 0), thickness=1,
                       max_width_px=600):
    if not text: return
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        (w_px, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if w_px > max_width_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur: lines.append(cur)

    x, y = org
    pad = 8
    box_w = min(max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines) + pad * 2,
                max_width_px + pad * 2)
    box_h = line_height * len(lines) + pad * 2
    cv2.rectangle(img, (x - 5, y - 22), (x - 5 + box_w, y - 22 + box_h), (0, 0, 0), -1)

    y_txt = y
    for l in lines:
        cv2.putText(img, l, (x + pad - 5, y_txt), font, font_scale, color, thickness, cv2.LINE_AA)
        y_txt += line_height


# =========================
# 모델 로드
# =========================
print("[VLM] Loading Qwen2-VL model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
    cache_dir=CACHE_DIR, offload_folder="offload_vlm"
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
print("[VLM] Loaded.")

# =========================
# Video Load (File)
# =========================
# 🟢 [수정] 파일에서 영상 읽기 준비 및 FPS 계산
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"[Video] Failed to open video file: {VIDEO_PATH}")

RS_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
RS_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🟢 [핵심 수정] 영상의 FPS를 읽어와서 대기 시간 계산 (정속 재생용)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0: video_fps = 30  # 읽기 실패 시 기본값
wait_msec = int(1000 / video_fps)  # 1000ms / FPS = 한 프레임당 대기 시간

# =========================
# 추론 워커
# =========================
infer_q = Queue(maxsize=1)
caption_lock = threading.Lock()
last_caption = ""
running = True

emg_counter = 0
last_emg_type = None
last_infer_time = None
vlm_fps = 0.0


def infer_worker():
    global last_caption, last_infer_time, vlm_fps
    global emg_counter, last_emg_type

    while running:
        try:
            bgr = infer_q.get(timeout=0.1)
        except Empty:
            continue
        try:
            pil_img = bgr_to_pil(bgr)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Always answer in English only."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text":
                        "- Describe in a COMPLETE Noun phrase within 10 tokens."
                        "⚠ RULES: "
                        "- Describe exactly what is visible in the image."
                        "- If the image contains a person or any part of a person, describe only that person or body part, and nothing else."
                        "- If a person is lying on the floor or ground, output exactly 'fallen'."
                        "- If a person is lying but not on the floor or ground (e.g., bed, sofa), output exactly 'lying'."
                     }
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(
                model.device)

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=10)
                gen = out[:, inputs.input_ids.shape[1]:]
                resp = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

            with caption_lock:
                last_caption = resp

            # 응급 상황 감지
            emg_type = detect_emergency(resp)
            if emg_type:
                print(f"[VLM] ⚠ Detected emergency: {emg_type}")
                if emg_type == last_emg_type:
                    emg_counter += 1
                else:
                    emg_counter = 1
                    last_emg_type = emg_type

                if emg_counter >= 3:
                    send_emergency(emg_type, 0.95)
                    emg_counter = 0
            else:
                emg_counter = 0
                last_emg_type = None

            now = time.time()
            if last_infer_time:
                vlm_fps = 1.0 / (now - last_infer_time)
            last_infer_time = now

        except Exception as e:
            with caption_lock:
                last_caption = f"(inference error: {e})"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            infer_q.task_done()


worker_th = threading.Thread(target=infer_worker, daemon=True)
worker_th.start()

# =========================
# 메인 루프
# =========================
prev_t = time.time()
last_enqueued = 0.0

try:
    while True:
        ret, bgr = cap.read()

        if not ret:
            break

        now = time.time()
        if (now - last_enqueued) >= INFER_EVERY_SEC and infer_q.empty():
            infer_q.put(bgr.copy())
            last_enqueued = now

        with caption_lock:
            cap_txt = last_caption
            cur_vlm_fps = vlm_fps

        overlay = bgr.copy()
        put_multiline_text(overlay, cap_txt, org=(12, 34), max_width_px=RS_WIDTH - 40)

        cv2.putText(overlay, f"VLM FPS: {cur_vlm_fps:4.2f}",
                    (RS_WIDTH - 150, RS_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Video + Qwen2-VL", overlay)

        # 🟢 [수정됨] 계산된 대기 시간 사용 (배속 재생 방지)
        if cv2.waitKey(wait_msec) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    running = False
    worker_th.join(timeout=1.0)
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()