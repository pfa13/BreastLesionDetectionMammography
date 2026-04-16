import torchvision

def get_model():
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")

    # 🔥 CLAVE REAL
    num_classes = 3  # 2 clases + background

    in_features = model.head.classification_head.num_classes

    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=9,
        num_classes=num_classes
    )

    return model