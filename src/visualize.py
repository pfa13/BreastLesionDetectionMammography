import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def show_predictions(model, dataset, device, num_images=5):
    model.eval()

    for i in range(num_images):
        img, target = dataset[i]

        img_tensor = img.to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)[0]

        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1,2,0).cpu())

        # GT
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                linewidth=2,
                edgecolor='g',
                facecolor='none'
            )
            ax.add_patch(rect)

            label_name = "mass" if label.item() == 1 else "calc"
            ax.text(x1, y1, label_name, color="green")

        # Predicciones
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score > 0.5:
                x1, y1, x2, y2 = box.cpu()
                rect = patches.Rectangle(
                    (x1, y1),
                    x2-x1,
                    y2-y1,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)

                label_name = "mass" if label.item() == 1 else "calc"
                ax.text(x1, y1, f"{label_name} {score:.2f}", color="red")

        plt.show()

