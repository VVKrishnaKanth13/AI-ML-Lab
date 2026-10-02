import matplotlib.pyplot as plt
from PIL import Image

from predict import predict_image


def visualize_prediction(image_path, top_k=5):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get predictions
    predictions = predict_image(
        image_path,
        top_k=top_k
    )

    # Create figure
    plt.figure(figsize=(8, 6))

    # Display image
    plt.imshow(image)
    plt.axis("off")

    # Create prediction text
    prediction_text = "\n".join(
        f"{rank}. {class_name}: {confidence * 100:.2f}%"
        for rank, (class_name, confidence)
        in enumerate(predictions, start=1)
    )

    # Display predictions as title
    plt.title(
        f"Top {top_k} Predictions\n\n{prediction_text}",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    image_path = "../images/dog.jpg"

    visualize_prediction(
        image_path,
        top_k=5
    )