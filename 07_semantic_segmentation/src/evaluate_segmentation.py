
import numpy as np


def calculate_iou(predicted_mask, ground_truth_mask, class_id):
    predicted = predicted_mask == class_id
    ground_truth = ground_truth_mask == class_id

    intersection = np.logical_and(
        predicted, ground_truth
    ).sum()

    union = np.logical_or(
        predicted, ground_truth
    ).sum()

    if union == 0:
        return None

    return intersection / union


def evaluate_all_classes(predicted_mask, ground_truth_mask):
    class_ids = np.union1d(
        np.unique(predicted_mask),
        np.unique(ground_truth_mask),
    )

    iou_scores = {}

    for class_id in class_ids:
        iou = calculate_iou(
            predicted_mask,
            ground_truth_mask,
            class_id,
        )

        iou_scores[int(class_id)] = iou

    return iou_scores

def calculate_mean_iou(iou_scores):
    valid_scores = [
        score
        for score in iou_scores.values()
        if score is not None
    ]

    if not valid_scores:
        return None

    return sum(valid_scores) / len(valid_scores)

def main():
    predicted_mask = np.array([
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 1],
    ])

    ground_truth_mask = np.array([
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ])

    iou_scores = evaluate_all_classes(
        predicted_mask,
        ground_truth_mask,
    )

    for class_id, iou in iou_scores.items():
        if iou is None:
            print(f"Class {class_id}: IoU undefined")
        else:
            print(f"Class {class_id}: IoU = {iou:.4f}")

    mean_iou = calculate_mean_iou(iou_scores)

    if mean_iou is None:
        print("Mean IoU: undefined")
    else:
        print(f"Mean IoU: {mean_iou:.4f}")



if __name__ == "__main__":
    main()
