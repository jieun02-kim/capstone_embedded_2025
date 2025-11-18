# =========================================================
#  Door Detection Training (YOLOv8n)
# =========================================================
from ultralytics import YOLO
import os
import yaml

# ---------------------------------------------------------


dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train/images")
val_dir   = os.path.join(dataset_dir, "val/images")
yaml_path = os.path.join(dataset_dir, "door_dataset.yaml")


data_yaml = {
    "train": train_dir,
    "val": val_dir,
    "nc": 1,                # 클래스 수
    "names": ["door"]       # 클래스 이름
}

os.makedirs(dataset_dir, exist_ok=True)
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"✅ YOLO dataset YAML saved -> {yaml_path}")

# ---------------------------------------------------------
model = YOLO("yolov8n.pt")   # 사전학습된 YOLOv8n 모델 사용


model.train(
    data=yaml_path,       # 데이터셋 yaml 경로
    epochs=50,            # 학습 epoch 수
    batch=16,             # 배치 크기
    imgsz=640,            # 입력 이미지 크기
    name="door_yolov8n",  # 결과 폴더 이름
    workers=2,            # 데이터로더 워커
    device=0              # GPU=0 / CPU는 'cpu'
)

# ---------------------------------------------------------

print("\n🎉 Training Complete!")
print("📂 Results saved in: runs/detect/door_yolov8n/")
print("🧠 Best weights: runs/detect/door_yolov8n/weights/best.pt")

# ---------------------------------------------------------
# 6️⃣ (선택) 훈련된 모델로 테스트
# ---------------------------------------------------------
# test_image = "door_test.jpg"
# results = model.predict(source=test_image, conf=0.5)
# results[0].show()
