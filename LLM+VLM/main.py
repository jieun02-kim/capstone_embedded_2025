# ======================================================================
# 🚫 Patch 1: Fake ALL_ATTENTION_FUNCTIONS for newer transformers
# ======================================================================
import sys, types, importlib
import os
import warnings
import logging

# 🔇 [최종] 모든 경고 및 지저분한 로그 강력 차단 설정
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "error"

# 1. 기본 경고 필터
warnings.filterwarnings("ignore")

# 2. Flash Attention 및 특정 문구 포함 경고 정밀 차단 (✅ 여기가 추가된 핵심입니다)
warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*numerical differences.*")
warnings.filterwarnings("ignore", category=UserWarning) # 모든 UserWarning 무시

try:
    mu = importlib.import_module("transformers.modeling_utils")
    if not hasattr(mu, "ALL_ATTENTION_FUNCTIONS"):
        mu.ALL_ATTENTION_FUNCTIONS = {}
        sys.modules["transformers.modeling_utils"] = mu
        # print("[patch] ...") # 패치 로그도 안 나오게 삭제함
except Exception as e:
    pass # 패치 실패 로그도 안 나오게 삭제함

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import json
import re
import socket
import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from transformers import BitsAndBytesConfig
import atexit
from gtts import gTTS
import pygame
import subprocess

try:
    import speech_recognition as sr
except ImportError:
    print("="*50)
    print("🚨 경고: 'SpeechRecognition' 라이브러리를 찾을 수 없습니다.")
    sr = None

# --- 로깅 및 경고 비활성화 (코드 1 기반) ---
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "error"

# 파이썬 경고 무시
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    StoppingCriteria,         # 👈 추가
    StoppingCriteriaList      # 👈 추가
)

# qwen_vl_utils.py 파일이 이 코드와 같은 위치에 있다고 가정합니다.
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("경고: qwen_vl_utils.py 파일을 찾을 수 없습니다. VLM 기능이 작동하지 않을 수 있습니다.")


    def process_vision_info(messages):
        # 임시 함수
        images = []
        for msg in messages:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        images.append(content['image'])
        return images, None

# ====================================================
# 0. 전역 설정
# ====================================================
# --- 모델 ID ---
# --- 모델 ID ---
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct" # ✅ 순수 텍스트 LLM으로 변경
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

# --- 통신 설정 ---
LLM_HOST, LLM_PORT = "127.0.0.1", 5555  # 프로젝트 2 (VLM)가 신호를 보낼 위치

# --- 전역 모델 변수 (load_models 함수에서 채워짐) ---
vlm_model, vlm_processor = None, None
llm_model, llm_tok = None, None

# --- (TTS 추가) 전역 TTS 엔진 변수 ---
# (tts_engine 변수 삭제, gTTS는 전역 객체가 필요 없음)
TTS_TEMP_FILE = "_temp_tts.mp3"
STT_TEMP_FILE = "_temp_stt.wav"
STT_SAMPLE_RATE = 44100
STT_DURATION = 10 # (4~5초 추천)

# [추가] 로봇이 말하는 중인지 확인하는 깃발 변수
is_speaking = False

# ====================================================
# 1. FSM 상태 및 이벤트 정의 (코드 1 기반)
# ====================================================

STATES = {"START", "IDLE", "FOLLOW", "GUIDE", "DELIVERY", "HOMING", "EMERGENCY"}

EVENTS = {
    "STARTUP_DONE",  # 부팅 완료
    "CMD_FOLLOW",  # "따라와"
    "CMD_GUIDE",  # "안내해줘"
    "CMD_DELIVERY",  # "배달해줘"
    "CMD_HOMING",  # "복귀해"
    "SWITCH_TO_GUIDE",  # (Follow 중) "이제 안내해줘"
    "SWITCH_TO_FOLLOW",  # (Guide 중) "이제 따라와"
    "GUIDE_DONE",  # 안내 완료 (도착 확인됨)
    "DELIVERY_DONE",  # 배달 완료 (도착 확인됨)
    "HOMING_DONE",  # 복귀 완료
    "GO_IDLE",  # "멈춰" 또는 "stop" (사용자 시뮬레이션 입력)
    "EMERGENCY_DETECTED",  # VLM (프로젝트 2)에서 감지
    "EMERGENCY_CLEARED",  # 응급 상태 수동 해제

    # --- 시뮬레이션을 위한 내부 이벤트 ---
    "ARRIVAL_SIGNAL",  # 로봇 도착 (사용자 시뮬레이션 입력 "arrived")
    "PATIENT_DETECTED",  # 환자 발견
    "LOCATION_RECEIVED",  # 현 위치 수신 (사용자 시뮬레이션 입력)

    # --- 실패 이벤트 ---
    "FOLLOW_FAILED",
    "GUIDE_FAILED",
    "DELIVERY_FAILED",
    "HOMING_FAILED"
}

# --- FSM 규칙 (우선순위 고려) ---
# (코드 1의 ALLOWED와 동일한 역할)
ALLOWED: Dict[str, set] = {
    "START": {"STARTUP_DONE"},
    "IDLE": {"CMD_FOLLOW", "CMD_GUIDE", "CMD_DELIVERY", "CMD_HOMING", "EMERGENCY_DETECTED"},
    "FOLLOW": {"SWITCH_TO_GUIDE", "GO_IDLE", "EMERGENCY_DETECTED", "FOLLOW_FAILED"},
    "GUIDE": {
        "SWITCH_TO_FOLLOW", "GO_IDLE", "EMERGENCY_DETECTED", "GUIDE_FAILED",
        "GUIDE_DONE", "LOCATION_RECEIVED", "ARRIVAL_SIGNAL"
    },
    "DELIVERY": {
        "GO_IDLE", "EMERGENCY_DETECTED", "DELIVERY_DONE", "DELIVERY_FAILED",
        "LOCATION_RECEIVED", "ARRIVAL_SIGNAL", "PATIENT_DETECTED"
    },
    "HOMING": {"GO_IDLE", "HOMING_DONE", "HOMING_FAILED", "EMERGENCY_DETECTED"},
    # EMERGENCY는 GO_IDLE (시뮬레이션 'stop') 또는 EMERGENCY_CLEARED로만 해제
    "EMERGENCY": {"GO_IDLE", "EMERGENCY_CLEARED"},
}


# ====================================================
# 2. 유틸리티 함수 (코드 1 기반)
# ====================================================

# (TTS 추가) 음성 출력 헬퍼 함수 (gTTS + playsound)
def speak(text: str):
    """ [수정됨] 말하는 동안 is_speaking 플래그를 True로 설정합니다. """
    global is_speaking  # 전역 변수 사용 선언

    print(f"[tts] {text}")

    if not pygame:
        return

    try:
        is_speaking = True  # 🚩 말하기 시작 (깃발 올림)

        tts = gTTS(text=text, lang='en')
        tts.save(TTS_TEMP_FILE)

        pygame.mixer.music.load(TTS_TEMP_FILE)
        pygame.mixer.music.play()

        # 재생이 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except Exception as e:
        print(f"[tts_error] gTTS/pygame 오류: {e}")
    finally:
        is_speaking = False  # 🏳️ 말 끝남 (깃발 내림) - 에러가 나도 무조건 실행됨

        # 파일 삭제 정리
        if os.path.exists(TTS_TEMP_FILE):
            try:
                time.sleep(0.5)
                os.remove(TTS_TEMP_FILE)
            except:
                pass

def active_emergency(sensors: dict) -> Tuple[Optional[str], float]:
    """ 센서 딕셔너리에서 활성화된 응급 상황을 확인합니다. """
    emg = sensors.get("emergency")  # ex: {"active": True, "type": "fall", ...}
    if not isinstance(emg, dict):
        return (None, 0.0)

    if emg.get("active", False):
        t = emg.get("type", "unknown")
        c = float(emg.get("confidence", 1.0))
        return (str(t), c)

    return (None, 0.0)


def go_idle_intent(utter: str) -> bool:
    """ 'stop' 명령어와 동일하게 처리될 'GO_IDLE' 의도를 빠르게 확인합니다. """
    s = re.sub(r"\s+", "", utter.lower())
    negatives = ("dontstop", "donotstop", "nohalt", "noidle", "neverstop",
                 "keepgoing", "continue", "resume", "dontcancel", "donotcancel")
    if any(n in s for n in negatives): return False
    keys = ("stop", "cancel", "halt", "pause", "idle", "abort", "quit", "end", "terminate")
    return any(k in s for k in keys)


def homing_intent(utter: str) -> bool:
    """ 홈 복귀 의도를 빠르게 확인합니다. """
    s = re.sub(r"\s+", "", utter.lower())
    keys = ("return", "dock", "charge", "charger", "chargingstation",
            "gohome", "backtobase", "goback", "gobacktodock", "gobackhome", "home")
    return any(k in s for k in keys)


# ====================================================
# 3. LLM 어댑터 (NLU) 정의 (코드 1 기반)
# ====================================================

class LLMAdapter:
    """
    LLM(Phi-4-mini)을 래핑하여 자연어(ASR) 입력을
    FSM 이벤트 JSON 제안으로 변환합니다.
    """

    def __init__(self):
        # LLM 로드 (전역 변수 사용)
        global llm_model, llm_tok
        if llm_model is None or llm_tok is None:
            raise ValueError("LLM이 로드되지 않았습니다. main의 load_models()를 먼저 호출해야 합니다.")

        self.model = llm_model
        self.tok = llm_tok

        # --- 시스템 프롬프트 (코드 1 기반) ---
        self.sys_prompt = (
            "You are a hospital robot command interpreter. "
            # ... (JSON 규칙) ...
            "Rules: "
            "- If event is CMD_GUIDE, params.destination (e.g., \"restroom\" or \"convenience store\") is REQUIRED. "
            "- If event is CMD_DELIVERY, params.room_id (e.g., \"patient room\") is REQUIRED. "
            "- speak must be a single-sentence confirmation/announcement. "

            # === 💡 [버그 수정] 구체적인 규칙 추가 ===
            "- If the user implies returning home (e.g., \"homing\", \"go home\", \"return to dock\", \"charge\"), always output CMD_HOMING. "  # ✅ Homing 규칙 추가
            "- If the user implies hunger (e.g., \"hungry\", \"food\", \"eat\"), always output CMD_GUIDE with destination \"convenience store\". "
            "- If the user implies needing a toilet (e.g., \"restroom\", \"bathroom\", \"poop\", \"pee\"), always output CMD_GUIDE with destination \"restroom\". "
            "- Only output CMD_DELIVERY if the user explicitly asks to \"deliver\" something to a specific room number. "
            # =======================================

            "- Do NOT invent extra top-level keys (e.g., reason/intent). "
            "- If user says 'stop/cancel/halt/pause/idle/abort/quit/end/terminate', always respond with "
            "{\"event\":\"GO_IDLE\",\"params\":{},\"speak\":\"Switching to idle mode.\",\"confidence\":0.99} ."
        )

    @staticmethod
    def _strip_to_json(text: str) -> str:
        """ LLM 응답에서 순수 JSON 문자열만 추출합니다. (코드 1 기반) """
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n?", "", s, count=1).rstrip()
            if s.endswith("```"):
                s = s[:-3].rstrip()

        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            # { }를 못찾으면, 혹시 <|end|> 토큰 등이 포함되었는지 확인
            eos_token = "<|end|>"
            if eos_token in s:
                s = s.split(eos_token)[0].strip()

            start, end = s.find("{"), s.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"JSON braces not found in text: {text}")

        s = s[start: end + 1].strip()
        return " ".join(s.splitlines()).strip()  # 여러 줄을 한 줄로

    def propose(self, asr_text: str, current_state: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        자연어 입력(asr_text)과 현재 상태(current_state)를 기반으로
        LLM에게 JSON 이벤트 생성을 요청합니다.
        """

        # 현재 상태에서 허용된 이벤트 목록을 힌트로 제공
        allowed = sorted(list(ALLOWED.get(current_state, set())))
        hint = {"current_state": current_state, "allowed_events": allowed, "context": context}
        hint_str = json.dumps(hint)

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": f"input(asr_text)='{asr_text}'.\nhint={hint_str}\nOutput ONE-LINE JSON only."},
        ]

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        # === 중요: pixel_values 같은 비전 입력 제거 ===
        if "pixel_values" in inputs:
            del inputs["pixel_values"]

        # === 텍스트 전용 추론 ===
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],  # ✅ 텍스트만
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=200,
                # temperature=0.0,
                do_sample=False,
            )

        text = self.tok.decode(out[0], skip_special_tokens=False)

        # <|assistant|> 태그 이후의 응답만 추출
        reply_part = text.split("<|assistant|>")
        if len(reply_part) > 1:
            reply = reply_part[-1]
        else:
            # 프롬프트 템플릿에 따라 응답 포맷이 다를 경우 대비
            reply = text.split(prompt)[-1] if prompt in text else text

        raw_json = self._strip_to_json(reply)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"--- LLM 원본 응답 (JSON 파싱 실패) ---")
            print(reply)
            print("-----------------------------------")
            raise SystemExit(f"JSON parse failed: {e}. Raw JSON: {raw_json}")

        # --- 기본값 채우기 (코드 1 기반) ---
        if isinstance(data, dict) and "event" not in data:
            for k in list(data.keys()):
                if isinstance(k, str) and k.isupper():
                    data = {"event": k}
                    break

        allowed_top = {"event", "params", "speak", "confidence", "notes"}
        data = {k: v for k, v in data.items() if k in allowed_top}
        data.setdefault("params", {})
        data.setdefault("confidence", 0.8)
        data.setdefault("speak", "")

        # --- 보정 로직 (코드 1 기반) ---
        if data.get("event") == "CMD_GUIDE":
            if not data["speak"]:
                d = data["params"].get("destination", "the destination")
                data["speak"] = f"Shall I guide you to {d}?"

        return data

# ====================================================
# 4. VLM/PDDL 파이프라인 정의 (코드 2 기반)
# ====================================================

class EmergencyStopper(StoppingCriteria):
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, input_ids, scores, **kwargs):
        # 응급 상태(EMERGENCY)가 되면 True를 반환 -> 생성 즉시 중단
        if self.runtime and self.runtime.state == "EMERGENCY":
            return True
        return False

class PDDLPipeline:
    """
    '코드 2'의 로직을 담당합니다.
    VLM으로 캡션을 생성하고, LLM으로 PDDL 초기 상태와 최종 명령을 생성합니다.
    """

    def __init__(self):
        # 모델 로드 (전역 변수 사용)
        global vlm_model, vlm_processor, llm_model, llm_tok
        if any(m is None for m in [vlm_model, vlm_processor, llm_model, llm_tok]):
            raise ValueError("VLM 또는 LLM이 로드되지 않았습니다. main의 load_models()를 먼저 호출해야 합니다.")

        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.llm_model = llm_model
        self.llm_tok = llm_tok

    def _run_vlm_captioning(self, image_dir: str, bbox_data: dict, runtime=None) -> Optional[dict]:
        """ [수정] 캡션을 아주 짧고 간결하게 생성하도록 변경 (로그 삭제 포함) """
        print(f"[PDDL_LOG] VLM 캡셔닝 시작")

        stopper = EmergencyStopper(runtime)
        stopping_criteria = StoppingCriteriaList([stopper])

        for obj_name, info in bbox_data.items():
            if runtime and runtime.state == "EMERGENCY": return None

            crop_path = os.path.join(image_dir, f"{obj_name}.png")

            # 파일 없으면 조용히 건너뜀
            if not os.path.exists(crop_path):
                info["caption"] = f"{obj_name} missing."
                continue

            try:
                image = Image.open(crop_path).convert("RGB")
                phrase = info.get("phrase", obj_name)

                # ✅ [핵심 수정] 프롬프트를 아주 강력하게 단축 (10단어 이하)
                prompt_text = (
                    f"Identify {phrase}. Read any text on it. "
                    f"Be extremely concise (max 10 words)."
                )

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]}
                ]

                text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = self.vlm_processor(text=[text], images=image_inputs, return_tensors="pt").to(
                    self.vlm_model.device)

                result_container = {"output": None}

                def generation_task():
                    try:
                        with torch.inference_mode():
                            result_container["output"] = self.vlm_model.generate(
                                **inputs,
                                max_new_tokens=30,  # ✅ [핵심 수정] 최대 길이를 50 -> 30으로 축소
                                stopping_criteria=stopping_criteria
                            )
                    except:
                        pass

                gen_thread = threading.Thread(target=generation_task)
                gen_thread.start()

                while gen_thread.is_alive():
                    gen_thread.join(timeout=0.1)
                    if runtime and runtime.state == "EMERGENCY": return None

                out = result_container["output"]
                if out is None: return None

                gen = out[:, inputs.input_ids.shape[1]:]
                caption = self.vlm_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

                if runtime and runtime.state == "EMERGENCY": return None

                bbox_data[obj_name]["caption"] = caption
                # 로그 확인용 (원하시면 주석 해제)
                print(f"[PDDL_LOG] 캡션 ({obj_name}): {caption}")

            except Exception:
                continue

        return bbox_data

    def _get_init_prompt(self, domain_name: str, domain_text: str, pddl_objects: str, bbox_data: dict) -> str:
        """ LLM에게 '초기 상태 PDDL' 생성을 요청하는 프롬프트를 만듭니다. (코드 2 기반) """
        prompt = f"""
Instruction:
You are generating the initial PDDL state (:init) for a hospital robot domain: {domain_name}.
Use the predicates and type rules defined in the given domain.

### Domain Definition:
{domain_text}

Follow these rules:
1. Use only predicates defined in the domain.
2. Include only information that can be **visually confirmed** from the scene (captions).
3. Do **NOT** include predicates that infer future actions or non-visual state (e.g., (following ...), (destination ...), (at ...)).
4. All objects in the PDDL objects list that are also described in the captions should be marked as (visible ...).
5. If the domain is 'guide' or 'delivery' and a caption mentions a specific label (e.g., "door labeled '19421'"), add (has_label <object_name> <label_name>).
6. Output must be valid PDDL syntax starting with "(:init" and ending with ")".

---
Q: [Domain: follow]
[Objects: (:objects patient1 - patient)]
[Caption: a patient wearing a hospital gown walking ahead.]
Write the initial state in PDDL? A:
(:init
    (visible patient1)
)

---
Q: [Domain: delivery]
[Objects: (:objects patient1 - patient)]
[Caption: a patient walking in the hallway.]
Write the initial state in PDDL? A:
(:init
    (visible patient1)
)

---
Q: [Domain: guide]
[Objects: (:objects door1 - door restroom - location)]
[Caption: a hospital room with a door labeled "Restroom"]
Write the initial state in PDDL? A:
(:init
    (visible door1)
    (has_label door1 Restroom)
)

---
Q: [Domain: {domain_name}]
[Objects: {pddl_objects}]
[Captions: {json.dumps(bbox_data, indent=2)}]
Write the initial state in PDDL? A:
"""
        return prompt

    def _get_command_prompt(self, domain_name: str, domain_text: str, pddl_objects: str, initial_state: str) -> str:
        """
        LLM에게 '최종 명령' 생성을 요청하는 프롬프트를 만듭니다. (신규 추가)
        '초기 상태'를 입력으로 받습니다.
        """
        prompt = f"""
Instruction:
You are a PDDL planner for the domain **'{domain_name}'**. Given the domain and the initial state, you **MUST STRICTLY** adhere to the **Domain Definition** provided below.
Determine which action's preconditions are met based **ONLY** on that definition.
Output ONLY the action name that should be executed.
If multiple actions are possible, choose the most relevant one.
If NO action's preconditions are met, output 'NO_ACTION'.
**DO NOT use actions from other domains.** For example, if the domain is 'follow', the action cannot be 'greet'.

Follow these specific domain rules:
1. If the domain is 'follow' and (visible patient) is true, the action MUST be 'follow'.
2. If the domain is 'follow' and (visible patient) is false, the action MUST be 'NO_ACTION'.

### Domain Definition:
{domain_text}

### Objects:
{pddl_objects}

### Initial State:
{initial_state}

---
Q: [Domain: follow]
[Initial State: (:init (visible patient1))]
What action's preconditions are met? A:
follow

---
Q: [Domain: follow]
[Initial State: (:init (not (visible patient1)))]
What action's preconditions are met? A:
NO_ACTION

---
Q: [Domain: delivery - arrive]
[Initial State: (:init (destination patient room) (visible door1) (has_label door1 patient room))]
What action's preconditions are met? A:
notice_arrived

---
Q: [Domain: guide - arrive]
[Initial State: (:init (destination restroom) (visible door) (has_label door restroom) (not (at restroom)))]
What action's preconditions are met? A:
arrived

---
Q: [Domain: delivery - greet]
[Initial State: (:init (visible patient1))]
What action's preconditions are met? A:
greet

---
Q: [Domain: {domain_name}]
[Initial State: {initial_state}]
What action's preconditions are met? A:
"""
        return prompt

    def _run_llm_generation(self, prompt_text: str, max_tokens: int) -> str:
        """ LLM(Phi-4-mini)을 실행하여 텍스트를 생성합니다. """
        prompt = self.llm_tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.llm_tok(prompt, return_tensors="pt").to(self.llm_model.device)
        eos_id = self.llm_tok.convert_tokens_to_ids("<|end|>") or self.llm_tok.eos_token_id

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                #temperature=0.0,
                do_sample=False,
                eos_token_id=eos_id
            )
            text_out = self.llm_tok.decode(outputs[0], skip_special_tokens=True)

        # 'A:' (답변) 부분만 추출
        result = text_out.split("A:")[-1].strip()

        # 혹시 <|end|> 토큰이 포함되었다면 제거
        if "<|end|>" in result:
            result = result.split("<|end|>")[0].strip()

        return result

    def execute_pipeline(self, domain_name: str, crop_dir: str, mock_json_file: str, object_pddl_file: str,
                         domain_pddl_file: str, destination_context: Optional[str] = None,
                         runtime=None) -> Tuple[str, str]:  # 👈 runtime 인자 확인

        # [체크 1] 시작 전 응급 확인
        if runtime and runtime.state == "EMERGENCY": return "ABORTED", ""

        # print(f"[PDDL_LOG] 파이프라인 시작: Domain={domain_name}")

        try:
            with open(mock_json_file, "r") as f:
                bbox_data = json.load(f)
            with open(object_pddl_file, "r") as f:
                pddl_objects = f.read().strip()
            with open(domain_pddl_file, "r") as f:
                domain_text = f.read().strip()
        except FileNotFoundError:
            return "ERROR_FILE_NOT_FOUND", ""

        # [체크 2] VLM 돌리기 전에 확인 (중복 제거됨)
        if runtime and runtime.state == "EMERGENCY": return "ABORTED", ""

        # --- 1. VLM 캡셔닝 ---
        # ✅ runtime을 인자로 넘겨줘서 내부 감시 시키기
        bbox_data_with_captions = self._run_vlm_captioning(crop_dir, bbox_data, runtime)

        # ✅ 내부에서 중단되어 None이 반환되었다면 즉시 종료
        if bbox_data_with_captions is None:
            return "ABORTED", ""

        # [체크 3] VLM 끝나고 LLM 넘어가기 전 응급 확인
        if runtime and runtime.state == "EMERGENCY": return "ABORTED", ""

        # --- 2. LLM 초기 상태 생성 ---
        init_prompt = self._get_init_prompt(domain_name, domain_text, pddl_objects, bbox_data_with_captions)
        initial_state = self._run_llm_generation(init_prompt, max_tokens=256)

        if not initial_state.startswith("(:init"):
            initial_state = f"(:init \n {initial_state} \n)"

        print(f"[PDDL_LOG] 1차 LLM 생성 (초기 상태):\n{initial_state}")

        # [체크 4] 최종 명령 생성 전 체크
        if runtime and runtime.state == "EMERGENCY": return "ABORTED", ""

        # --- 3. 목적지/상황 컨텍스트 추가 ---
        final_initial_state = initial_state
        if destination_context:
            final_initial_state = final_initial_state.rstrip(")") + f" {destination_context} )"
            if domain_name == "guide" and "destination" in destination_context:
                dest_name = destination_context.split()[-1].rstrip(")")
                final_initial_state = final_initial_state.rstrip(")") + f" (not (at {dest_name})) )"

        # --- 4. LLM 최종 명령 생성 ---
        command_prompt = self._get_command_prompt(domain_name, domain_text, pddl_objects, final_initial_state)
        final_command_raw = self._run_llm_generation(command_prompt, max_tokens=50)
        final_command = final_command_raw.splitlines()[0].strip()

        # [체크 5] 마지막 리턴 직전 체크
        if runtime and runtime.state == "EMERGENCY": return "ABORTED", ""

        print(f"[PDDL_LOG] 2차 LLM 생성 (최종 명령): {final_command}")
        return final_command, final_initial_state

# ====================================================
# 5. FSM 런타임 (Gatekeeper 및 Dispatcher)
# ====================================================

# --- FSM 런타임 상태 객체 ---
@dataclass
class RobotRuntime:
    """ FSM의 현재 상태와 로봇의 동작을 관리합니다. """
    state: str = "START"
    last_state: str = "START"
    home_pose: str = "dock-A"
    last_destination: Optional[str] = None
    last_room: Optional[str] = None

    # ✅ [수정] 이동 상태 및 단계(Step) 변수 추가
    simulating_travel: bool = False
    simulation_step: int = 0  # 0:정지, 1:환자만나는중, 2:도착하는중

    # MOCK_PATHS는 기존과 동일
    MOCK_PATHS = {
        "follow": ("crop/follow_crop", "crop/follow_crop/follow.json", "crop/follow_crop/object.pddl",
                   "domain/follow_domain.pddl"),
        "guide": (
        "crop/guide_crop", "crop/guide_crop/guide.json", "crop/guide_crop/object.pddl", "domain/guide_domain.pddl"),
        "delivery_greet": ("crop/delivery_crop/before_arrive", "crop/delivery_crop/before_arrive/before_arrive.json",
                           "crop/delivery_crop/before_arrive/object.pddl", "domain/delivery_domain.pddl"),
        "delivery_arrive": ("crop/delivery_crop/after_arrive", "crop/delivery_crop/after_arrive/after_arrive.json",
                            "crop/delivery_crop/after_arrive/object.pddl", "domain/delivery_domain.pddl"),
    }

    def transition(self, event: str) -> None:
        """ FSM 상태를 전이시킵니다. """
        prev = self.state

        if self.state == "START" and event == "STARTUP_DONE":
            self.state = "IDLE"
        elif self.state == "IDLE":
            if event == "CMD_FOLLOW":
                self.state = "FOLLOW"
            elif event == "CMD_GUIDE":
                self.state = "GUIDE"
            elif event == "CMD_DELIVERY":
                self.state = "DELIVERY"
            elif event == "CMD_HOMING":
                self.state = "HOMING"

        elif self.state == "FOLLOW":
            if event == "SWITCH_TO_GUIDE":
                self.state = "GUIDE"
            elif event in {"GO_IDLE", "FOLLOW_FAILED"}:
                self.state = "IDLE"

        elif self.state == "GUIDE":
            if event == "SWITCH_TO_FOLLOW":
                self.state = "FOLLOW"
            elif event in {"GO_IDLE", "GUIDE_FAILED", "GUIDE_DONE"}:
                self.state = "IDLE"

        elif self.state == "DELIVERY":
            if event in {"GO_IDLE", "DELIVERY_FAILED", "DELIVERY_DONE"}:
                self.state = "IDLE"
            elif event == "CMD_HOMING":
                self.state = "HOMING"

        elif self.state == "HOMING":
            if event in {"GO_IDLE", "HOMING_FAILED", "HOMING_DONE"}: self.state = "IDLE"

        # --- 글로벌 인터럽트 ---
        if event == "EMERGENCY_DETECTED" and self.state != "EMERGENCY":
            self.state = "EMERGENCY"
            # ✅ 응급 시 시뮬레이션 변수 초기화
            self.simulating_travel = False
            self.simulation_step = 0
        elif self.state == "EMERGENCY":
            if event in {"EMERGENCY_CLEARED", "GO_IDLE"}: self.state = "IDLE"

        if prev != self.state:
            print(f"[state] {prev} -> {self.state}")
            self.last_state = prev

    def dispatch(self, decision: Dict[str, Any], pddl_pipe: 'PDDLPipeline'):
        """ [최종] 모든 모드에서 응급 상황 시 즉시 중단(Abort) 및 로그 순서 보장 """
        ev = decision["event"]
        params = decision.get("params", {})
        speak_text = decision.get("speak", "")

        # 1. 상태 전이
        self.transition(ev)

        # [순서 보장 1] 응급 로그를 최우선 출력
        if ev == "EMERGENCY_DETECTED":
            emg_type = params.get("type", "unknown")
            print(f"[action] 🚨 EMERGENCY! broadcasting alert! type={emg_type}")

        # 2. TTS 실행
        if speak_text:
            speak(speak_text)

        # [안전장치 1]
        if self.state == "EMERGENCY" and ev != "EMERGENCY_DETECTED":
            return

        # 3. 행동 실행 (Execute)

        # --- CASE A: Follow ---
        if ev == "CMD_FOLLOW":
            print(f"[action] running follow check pipeline...")
            paths = self.MOCK_PATHS["follow"]
            command, _ = pddl_pipe.execute_pipeline("follow", paths[0], paths[1], paths[2], paths[3], None,
                                                    runtime=self)

            if command == "ABORTED" or self.state == "EMERGENCY": return

            if command.lower() == "follow":
                print(f"[action] send action to robot : follow")
                print(f"--- (SYSTEM) 로봇이 따라가기 시작합니다. 중단하려면 'stop' 입력 ---")
            else:
                print(f"[action] follow check failed. returning to idle.")
                speak("I cannot see anyone.")
                self.transition("FOLLOW_FAILED")

        # --- CASE B: Guide ---
        elif ev == "CMD_GUIDE":
            self.last_destination = params["destination"]
            print(f"[action] destination set: {self.last_destination}")
            print(f"[action] send destination to robot: {self.last_destination}")
            print(f"--- (SYSTEM) 로봇이 목적지로 이동합니다... ---")

            # Guide는 중간 단계 없이 바로 도착으로 설정
            self.simulating_travel = True
            self.simulation_step = 2

            # --- CASE C: Delivery (✅ 누락된 부분 추가됨) ---
        elif ev == "CMD_DELIVERY":
            self.last_room = params["room_id"]
            print(f"[action] destination set: {self.last_room}")
            print(f"[action] send destination to robot: {self.last_room}")
            print(f"--- (SYSTEM) 로봇이 배달 장소로 이동합니다... ---")

            # ✅ Delivery는 1단계(환자 만남)부터 시작
            self.simulating_travel = True
            self.simulation_step = 1

        # --- CASE D: Greeting (Patient) ---
        elif ev == "PATIENT_DETECTED":
            if self.state != "DELIVERY": return
            print(f"[action] patient detected on the way...")

            paths = self.MOCK_PATHS["delivery_greet"]
            command, initial_state = pddl_pipe.execute_pipeline("delivery - greet", paths[0], paths[1], paths[2],
                                                                paths[3], None, runtime=self)

            if command == "ABORTED" or self.state == "EMERGENCY": return

            if command.lower() == "greet":
                greet_target = "Patient"
                print(f"[action] send action to robot : greet (Target: {greet_target})")
                speak(f"Hello, how are you feeling?")
            else:
                print(f"[action] greet failed (no patient found)")

            # ✅ 인사 후 다시 이동 (2단계로 진입)
            print(f"--- (SYSTEM) 인사를 마쳤습니다. 다시 목적지로 이동합니다... ---")
            self.simulating_travel = True
            self.simulation_step = 2

        # --- CASE E: Arrival Check ---
        elif ev == "ARRIVAL_SIGNAL":
            # 도착했으므로 시뮬레이션 종료
            self.simulating_travel = False
            self.simulation_step = 0

            if self.state == "GUIDE":
                paths = self.MOCK_PATHS["guide"]
                context = f"(destination {self.last_destination})"
                command, _ = pddl_pipe.execute_pipeline("guide", paths[0], paths[1], paths[2], paths[3], context,
                                                        runtime=self)

                if command == "ABORTED" or self.state == "EMERGENCY": return

                if command.lower() == "arrived":
                    print(f"[action] arrived at destination.")
                    self.transition("GUIDE_DONE")
                else:
                    print(f"[action] wrong location.")
                    self.transition("GUIDE_FAILED")

            elif self.state == "DELIVERY":
                paths = self.MOCK_PATHS["delivery_arrive"]
                context = f"(destination {self.last_room})"
                command, _ = pddl_pipe.execute_pipeline("delivery - arrive", paths[0], paths[1], paths[2], paths[3],
                                                        context, runtime=self)

                if command == "ABORTED" or self.state == "EMERGENCY": return

                if command.lower() == "notice_arrived":
                    print(f"[action] arrived. Delivery complete.")
                    self.transition("DELIVERY_DONE")
                else:
                    print(f"[action] wrong location.")
                    self.transition("DELIVERY_FAILED")

        elif ev == "GO_IDLE":
            print("[action] stopping all tasks. returning to idle.")
            self.simulating_travel = False
            self.simulation_step = 0

        elif ev == "EMERGENCY_CLEARED":
            print("[action] emergency cleared.")
            self.simulating_travel = False
            self.simulation_step = 0
            self.transition("GO_IDLE")

# --- Gatekeeper (코드 1 기반) ---
def gatekeep(current_state: str, p: Dict[str, Any], sensors: Dict[str, Any], rt: RobotRuntime) -> Dict[str, Any]:
    """ LLM 제안(p)을 FSM 규칙과 센서 상태에 따라 검증(validate)합니다. """

    # 1. 응급 상황 최우선 처리
    emg_type, emg_conf = active_emergency(sensors)
    if emg_type and current_state not in {"START", "EMERGENCY"}:
        return {
            "ok": True, "event": "EMERGENCY_DETECTED",
            "params": {"type": emg_type, "confidence": emg_conf},
            "speak": f"Emergency detected: {emg_type}!", "confidence": 1.0,
        }

    # 2. 기본 스키마 검증
    if not isinstance(p, dict) or "event" not in p:
        return {"ok": False, "error": "bad_schema"}

    ev = p["event"]
    if ev not in EVENTS:
        return {"ok": False, "error": "unknown_event", "event": ev}

    # 3. FSM 규칙(ALLOWED) 검증
    if ev not in ALLOWED.get(current_state, set()):
        return {"ok": False, "error": "not_allowed", "state": current_state, "event": ev}

    # 4. 파라미터 누락 검증
    if ev == "CMD_GUIDE" and not p.get("params", {}).get("destination"):
        return {"ok": False, "error": "need_destination"}
    if ev == "CMD_DELIVERY" and not p.get("params", {}).get("room_id"):
        return {"ok": False, "error": "need_room_id"}

    # 5. (시뮬레이션) 'stop' 입력은 항상 GO_IDLE로
    if ev == "GO_IDLE":
        p["confidence"] = 1.0  # 'stop'은 항상 100% 신뢰

    # (코드 1의 배터리/신뢰도 검증 등은 데모를 위해 생략)

    return {"ok": True, **p}  # 모든 검사 통과


# ====================================================
# 6. 응급 서버 (코드 1 기반)
# ====================================================

def start_emergency_server(rt: RobotRuntime, sensors: dict, host=LLM_HOST, port=LLM_PORT):
    """ (프로젝트 2) VLM의 응급 신호를 수신하는 서버 스레드 """
    print(f"[VLM_Server] 응급 신호 리스너 시작 ({host}:{port})")

    def handler():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, port))
            srv.listen(1)
            while True:
                conn, _ = srv.accept()
                try:
                    data = conn.recv(4096).decode("utf-8").strip()
                    if not data: continue

                    msg = json.loads(data)
                    emg_type = msg.get("type")
                    conf = msg.get("confidence", 0.9)

                    if emg_type and rt.state not in {"START", "EMERGENCY"}:
                        print(f"\n[VLM_Server] 🚨 VLM (Project 2)에서 응급 신호 수신: {msg}")

                        # 센서 상태 업데이트
                        sensors["emergency"] = {"active": True, "type": emg_type, "confidence": conf}
                        # FSM 강제 전환
                        dec = {
                            "event": "EMERGENCY_DETECTED",
                            "params": {"type": emg_type},
                            "speak": f"Emergency detected: {emg_type}!",
                            "confidence": 1.0,
                        }
                        # (중요) 메인 스레드와 충돌을 피하기 위해
                        # 실제로는 큐(Queue)를 사용해야 하지만, 데모에서는 직접 호출
                        rt.dispatch(dec, pddl_pipeline)  # pddl_pipeline은 main의 전역 객체
                except Exception as e:
                    print(f"[VLM_Server] 오류: {e} (data: {data})")
                finally:
                    conn.close()
        except OSError as e:
            print(f"[VLM_Server] 소켓 오류 (포트 사용 중?): {e}")
        finally:
            srv.close()

    th = threading.Thread(target=handler, daemon=True)
    th.start()


# ====================================================
# 7. 메인 실행 함수
# ====================================================

# (전역) PDDL 파이프라인 객체. 응급 서버 스레드에서도 접근해야 함.
pddl_pipeline: Optional[PDDLPipeline] = None

# --- ✅ [STT 수정] ---
recognizer: Optional[sr.Recognizer] = None
# microphone 변수 삭제
# --- ✅ [STT 수정 끝] ---

def initialize_stt():
    """ STT Recognizer를 초기화합니다. (arecord 사용) """
    global recognizer
    if not sr:  # ✅ [수정] sr만 확인
        print("🚨 STT 라이브러리 누락. 'pip install SpeechRecognition'")
        return False

    try:
        recognizer = sr.Recognizer()  # (유지)

        # --- ✅ [STT 수정] ---
        # (sounddevice 관련 코드 모두 삭제)
        # 'arecord'가 존재하는지 간단히 확인 (선택적)
        try:
            subprocess.run(["arecord", "-l"], capture_output=True, timeout=2)
            print("[STT] Recognizer 초기화 완료.")
            print("[STT] 'arecord' (ALSA) 녹음기 사용 준비 완료.")
        except (FileNotFoundError, subprocess.TimeoutExpiredError):
            raise Exception("'arecord' 명령어를 찾을 수 없거나 응답이 없습니다. 'sudo apt-get install alsa-utils'를 실행하세요.")
        # --- ✅ [STT 수정 끝] ---

        return True
    except Exception as e:
        print("=" * 50)
        print(f"🚨 STT/arecord 초기화 실패: {e}")
        print("=" * 50)
        return False


def listen_for_command(sensors: dict, current_state: str) -> Optional[str]:
    """ (수정됨) beepy 없음, Interrupt 로그 삭제, 응급 시 조용히 중단 """
    if not recognizer:
        print("[STT_Error] STT가 초기화되지 않았습니다.")
        return None

    print(f"\n🎤 말해주세요...")

    arecord_cmd = [
        "arecord", "-D", "pulse", "-f", "S16_LE",
        "-r", str(STT_SAMPLE_RATE), "-d", str(STT_DURATION),
        STT_TEMP_FILE
    ]

    try:
        # 녹음 프로세스 시작 (Non-blocking)
        proc = subprocess.Popen(arecord_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start_time = time.time()

        while proc.poll() is None:
            # 응급 센서가 켜졌는지 확인
            is_emergency_sensor = sensors.get("emergency", {}).get("active", False)

            # "센서가 켜졌고" AND "현재 상태가 아직 응급이 아닐 때"만 중단
            if is_emergency_sensor and current_state != "EMERGENCY":
                proc.terminate()
                proc.wait()
                # 🗑️ [Interrupt] 문구 삭제함 (조용히 리턴)
                if os.path.exists(STT_TEMP_FILE): os.remove(STT_TEMP_FILE)
                return None

            if time.time() - start_time > STT_DURATION + 1:
                proc.terminate()
                break
            time.sleep(0.1)

        if not os.path.exists(STT_TEMP_FILE): return None

        with sr.AudioFile(STT_TEMP_FILE) as source:
            audio_data = recognizer.record(source)

        print("[STT] 인식 중...")
        text = recognizer.recognize_google(audio_data, language='en-US')
        print(f"User (STT)> {text}")
        return text.lower().strip()

    except Exception as e:
        # 에러 발생 시 조용히 넘어감
        return None
    finally:
        if os.path.exists(STT_TEMP_FILE):
            try: os.remove(STT_TEMP_FILE)
            except: pass

# --- ✅ [STT 추가 끝] ---

def load_models():
    """ VLM과 LLM 모델을 전역 변수에 로드합니다. """
    global vlm_model, vlm_processor, llm_model, llm_tok

    # 1. VLM (Qwen2-VL)을 CPU로 로드
    print("[Main] VLM (Qwen2-VL) 로드 중... (CPU로 오프로드)")
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        cache_dir=CACHE_DIR
    )
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, cache_dir=CACHE_DIR)

    # 2. LLM (Phi-3-mini, NLU+PDDL)도 CPU로 로드
    print("[Main] LLM (Phi-3-mini) 로드 중... (GPU 4bit, low VRAM mode)")
    llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",  # → 자동으로 GPU로 올림
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # Phi-3는 VLM 패치가 필요 없으므로 삭제된 상태 유지
    print("[Main] 모든 모델 로드 완료 ✅")


def main():
    print("=== 🤖 헬스케어 로봇 FSM + PDDL 시뮬레이터 ===")

    try:
        load_models()
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
        print("Huggingface 토큰 또는 네트워크 연결을 확인하세요.")
        return

    print("[Main] TTS (gTTS) 사용 준비 완료.")

    # --- ✅ [pygame 패치] ---
    print("[Main] Pygame Mixer (오디오) 초기화 중...")
    pygame.mixer.init()
    # --- ✅ [패치 끝] ---

    # --- ✅ [STT 추가] STT 초기화 ---
    if not initialize_stt():
        print("🚨 치명적 오류: STT(마이크)를 시작할 수 없어 프로그램을 종료합니다.")
        return
    # --- ✅ [STT 추가 끝] ---

    # 1. 전역 파이프라인 객체 생성 (코드 2 로직)
    global pddl_pipeline
    pddl_pipeline = PDDLPipeline()

    # 2. FSM 런타임 객체 생성 (코드 1 로직)
    rt = RobotRuntime(state="START")

    # 3. LLM NLU 어댑터 생성 (코드 1 로직)
    llm_adapter = LLMAdapter()

    # 4. 센서 (시뮬레이션용)
    sensors = {"battery": 95, "emergency": {"active": False}}

    # 5. VLM 응급 서버 시작
    start_emergency_server(rt, sensors)

    # 6. 시스템 부팅
    rt.dispatch({"event": "STARTUP_DONE", "speak": "System online. Waiting for command."}, pddl_pipeline)

    # 7. 메인 루프 (음성 입력)
    while True:
        # 🛑 [1] 말하는 중 대기
        if is_speaking:
            while is_speaking: time.sleep(0.1)

        # 🛑 [2] 응급 메시지
        if rt.state == "EMERGENCY":
            print(f"--- (SYSTEM) 응급 상태입니다. 해제하려면 'clear emergency' 또는 'stop' 입력 ---")

        # ==========================================================
        # ✅ [수정] 단계별 시뮬레이션 로직 (Step 1 -> Step 2)
        # ==========================================================
        line = None

        if rt.simulating_travel and rt.state != "EMERGENCY":
            # 30초 대기 (이동 시뮬레이션)
            time.sleep(30.0)

            # 배달 중이고 1단계(출발 직후)라면 -> 환자 만남 신호 발생
            if rt.state == "DELIVERY" and rt.simulation_step == 1:
                print("\n[Robot] (Simulated) Patient detected on path.")
                line = "patient"

            # 배달 중 2단계(인사 후) 또는 그 외(Guide) -> 도착 신호 발생
            else:
                print("\n[Robot] location clear")
                line = "location clear"
        else:
            try:
                line = listen_for_command(sensors, rt.state)
            except (EOFError, KeyboardInterrupt):
                print("\n[EXIT] (Ctrl+C로 종료)");
                break

        # 🛑 [3] 처리 로직
        if line is None:
            time.sleep(1.0)
            continue

        if line.lower() in {"exit", "quit"}:
            print("[EXIT]");
            break

        proposal = {}

        # --- 7A. 시뮬레이션/로봇 신호 처리 ---
        if go_idle_intent(line):
            proposal = {"event": "GO_IDLE", "speak": "Stopping current task."}
            if rt.state == "EMERGENCY":
                sensors["emergency"]["active"] = False
                proposal["speak"] = "Emergency cleared. Returning to idle."

        elif line.lower() in {"location clear", "location cleared", "location here"}:
            proposal = {"event": "ARRIVAL_SIGNAL"}

        # ✅ 환자 입력 처리
        elif line.lower() == "patient":
            proposal = {"event": "PATIENT_DETECTED"}

        elif line.lower() == "nurse":
            proposal = {"event": "NURSE_DETECTED"}
        elif line.lower() == "doctor":
            proposal = {"event": "DOCTOR_DETECTED"}

        # --- 7B. (NLU) 자연어 명령 처리 ---
        else:
            try:
                # LLM (Phi-4) 호출
                proposal = llm_adapter.propose(line, rt.state, sensors)
                #print(f"[llm_nlu] {proposal.get('event')}, dest={proposal.get('params', {}).get('destination') or proposal.get('params', {}).get('room_id')}")
            except Exception as e:
                print(f"[llm_nlu] 🚨 NLU 어댑터 오류: {e}")
                continue

        # --- 7C. Gatekeep & Dispatch ---
        if not proposal: continue

        # (Gatekeeper가 proposal을 검증하고, 필요시 응급 이벤트로 덮어씀)
        decision = gatekeep(rt.state, proposal, sensors, rt)

        if decision["ok"]:
            # (Dispatch가 FSM 상태를 변경하고 PDDL 파이프라인 등 '행동'을 실행)
            rt.dispatch(decision, pddl_pipeline)
        else:
            print(f"[fsm_error] {decision['error']} (state={rt.state}, event={proposal.get('event')})")


if __name__ == "__main__":
    # (qwen_vl_utils.py가 필요할 수 있음)
    main()