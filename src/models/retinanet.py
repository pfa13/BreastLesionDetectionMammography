import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def get_model(num_classes):

    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")

    # número de anchors por location (depende del backbone FPN)
    num_anchors = model.head.classification_head.num_anchors

    # sustituir correctamente la head
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes  # ⚠️ SIN +1 (RetinaNet ya maneja background)
    )

    return model