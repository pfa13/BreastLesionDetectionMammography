from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolov8n.pt")  # 🔥 nano (más rápido)

    model.train(
        data="data.yaml",
        imgsz=512,   # 🔥 antes 1024
        epochs=5,
        batch=2
    )