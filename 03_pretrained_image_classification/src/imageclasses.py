from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
class_names = weights.meta["categories"]

with open("../imagenet_classes.txt", "w", encoding="utf-8") as f:
    for class_name in class_names:
        f.write(class_name + "\n")

print(f"Saved {len(class_names)} ImageNet classes")