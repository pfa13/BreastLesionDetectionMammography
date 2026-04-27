from ultralytics import YOLO

def get_model():
    return YOLO("rtdetr-l.pt")