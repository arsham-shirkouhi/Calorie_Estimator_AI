from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .model import create_efficientnet_b0

NUTRITION_COLUMNS = ["calories", "protein", "carbs", "fat", "fiber", "sugar", "sodium"]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[1] / p).resolve()


def _load_class_names(nutrition_csv_path: str | Path) -> List[str]:
    nutrition_df = pd.read_csv(_resolve_path(nutrition_csv_path))
    return nutrition_df["food"].astype(str).tolist()


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_model(model_path: str | Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = create_efficientnet_b0(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(_resolve_path(model_path), map_location=device)

    # Support either plain state_dict or full checkpoint dictionary.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def predict_food(
    image_path: str | Path,
    model_path: str | Path = "models/best_model.pth",
    nutrition_csv_path: str | Path = "data/nutrition_database.csv",
) -> Tuple[str, float]:
    """
    Run EfficientNet-B0 inference and return predicted food class + confidence.

    Returns:
        (predicted_food_class, confidence_percent)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = _load_class_names(nutrition_csv_path)
    model = _load_model(model_path, num_classes=len(class_names), device=device)

    image = Image.open(_resolve_path(image_path)).convert("RGB")
    image_tensor = _build_transform()(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence_percent = float(confidence.item() * 100.0)
    return predicted_class, confidence_percent


def get_nutrition(
    predicted_food_class: str, nutrition_csv_path: str | Path = "data/nutrition_database.csv"
) -> Dict[str, float]:
    """
    Look up nutrition values for a predicted class.

    Returns:
        Dictionary with calories, protein, carbs, fat, fiber, sugar, sodium (per 100g).
    """
    nutrition_df = pd.read_csv(_resolve_path(nutrition_csv_path))
    match = nutrition_df.loc[nutrition_df["food"] == predicted_food_class]

    if match.empty:
        raise ValueError(f"Food class '{predicted_food_class}' not found in nutrition database.")

    row = match.iloc[0]
    return {col: float(row[col]) for col in NUTRITION_COLUMNS}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run food prediction and nutrition lookup.")
    parser.add_argument("images", nargs="+", help="One or more input image paths.")
    parser.add_argument("--model-path", default="models/best_model.pth", help="Path to model .pth file.")
    parser.add_argument(
        "--nutrition-csv",
        default="data/nutrition_database.csv",
        help="Path to nutrition CSV file.",
    )
    args = parser.parse_args()

    for image_path in args.images:
        predicted_class, confidence = predict_food(
            image_path=image_path, model_path=args.model_path, nutrition_csv_path=args.nutrition_csv
        )
        nutrition = get_nutrition(predicted_class, nutrition_csv_path=args.nutrition_csv)

        print(f"image: {image_path}")
        print(f"predicted class: {predicted_class}")
        print(f"confidence: {confidence:.2f}%")
        for col in NUTRITION_COLUMNS:
            print(f"{col}: {nutrition[col]}")
        print("-" * 40)


if __name__ == "__main__":
    _cli()
