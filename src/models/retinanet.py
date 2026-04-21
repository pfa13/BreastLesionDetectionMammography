import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn


def get_model(num_classes=3):

    model = retinanet_resnet50_fpn(weights="DEFAULT")

    # torchvision YA espera num_classes en training loop
    # SOLO cambia predictor correctamente:

    from torchvision.models.detection.retinanet import RetinaNetHead

    num_classes = num_classes + 1  # background

    # ✔ forma segura: reemplazar head completa
    model.head = RetinaNetHead(
        in_channels=256,
        num_anchors=model.head.classification_head.num_anchors,
        num_classes=num_classes
    )

    return model