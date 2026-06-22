import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
PRIVATE_RUNS_DIR = BASE_DIR / "var" / "runs"
MODEL_CACHE_DIR = BASE_DIR / "var" / "models"
PUBLIC_RESULTS_DIR = BASE_DIR / "static" / "runs"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


def load_yolo_model(model_file):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run `pip install -r requirements.txt`.") from exc

    model_path = None
    if model_file and model_file.filename:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_CACHE_DIR / secure_filename(model_file.filename)
        model_file.save(model_path)
    else:
        env_path = os.environ.get("MODEL_PATH")
        if env_path:
            model_path = Path(env_path)

    if not model_path:
        raise RuntimeError("Upload a YOLO .pt model or set MODEL_PATH before running evaluation.")
    if not Path(model_path).exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    return YOLO(str(model_path))


def clean_previous_runs():
    for directory in (PRIVATE_RUNS_DIR, PUBLIC_RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def parse_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    annotations_by_image = defaultdict(list)

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image_id = ann.get("image_id")
        if image_id not in images or "bbox" not in ann:
            continue
        x, y, w, h = ann["bbox"]
        annotations_by_image[image_id].append(
            {
                "category_id": ann["category_id"],
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
            }
        )

    if not images:
        raise ValueError("COCO JSON does not contain images.")
    if not categories:
        raise ValueError("COCO JSON does not contain categories.")

    return images, categories, annotations_by_image


def save_uploaded_images(files, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_by_basename = {}

    for file in files:
        if not file or not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        filename = secure_filename(Path(file.filename).name)
        if not filename:
            continue
        path = target_dir / filename
        file.save(path)
        saved_by_basename[filename] = path

    if not saved_by_basename:
        raise ValueError("Upload at least one image file.")

    return saved_by_basename


def resolve_image_paths(coco_images, saved_by_basename):
    resolved = {}
    missing = []

    for image_id, image_info in coco_images.items():
        file_name = image_info.get("file_name", "")
        basename = Path(file_name).name
        if basename in saved_by_basename:
            resolved[image_id] = saved_by_basename[basename]
        else:
            missing.append(file_name)

    return resolved, missing


def category_mapper(model_names, categories):
    if isinstance(model_names, list):
        model_names = {index: name for index, name in enumerate(model_names)}

    category_by_name = {normalize_label(name): cat_id for cat_id, name in categories.items()}
    model_name_by_class = {
        int(class_id): normalize_label(name) for class_id, name in model_names.items()
    }
    unmatched = sorted(
        {
            str(model_names[class_id])
            for class_id, normalized_name in model_name_by_class.items()
            if normalized_name not in category_by_name
        }
    )
    if unmatched:
        expected = ", ".join(categories.values())
        found = ", ".join(str(model_names[class_id]) for class_id in sorted(model_name_by_class))
        raise ValueError(
            "Model class names must match COCO category names before mAP can be trusted. "
            f"COCO categories: {expected}. Model classes: {found}. "
            f"Unmatched model classes: {', '.join(unmatched)}."
        )

    def map_class(model_class_id):
        normalized_name = model_name_by_class.get(model_class_id)
        if normalized_name is None:
            raise ValueError(f"Prediction uses unknown model class id {model_class_id}.")
        return category_by_name[normalized_name]

    return map_class


def normalize_label(value):
    return str(value).strip().casefold()


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def average_precision(recalls, precisions):
    ap = 0.0
    for recall_threshold in [x / 100 for x in range(101)]:
        precision_at_recall = [
            precision for recall, precision in zip(recalls, precisions) if recall >= recall_threshold
        ]
        ap += max(precision_at_recall) if precision_at_recall else 0.0
    return ap / 101


def compute_ap_for_class(predictions, ground_truths, iou_threshold):
    total_gt = sum(len(items) for items in ground_truths.values())
    if total_gt == 0:
        return None

    matched = {image_id: [False] * len(items) for image_id, items in ground_truths.items()}
    sorted_predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)

    true_positives = []
    false_positives = []

    for prediction in sorted_predictions:
        image_id = prediction["image_id"]
        candidates = ground_truths.get(image_id, [])
        image_matched = matched.setdefault(image_id, [False] * len(candidates))
        best_iou = 0.0
        best_index = None

        for index, gt_box in enumerate(candidates):
            if image_matched[index]:
                continue
            iou = box_iou(prediction["bbox"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            image_matched[best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    recalls = []
    precisions = []
    cumulative_tp = 0
    cumulative_fp = 0

    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / total_gt)
        precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))

    return average_precision(recalls, precisions)


def evaluate_map(predictions, ground_truths, categories):
    thresholds = [round(0.5 + index * 0.05, 2) for index in range(10)]
    class_rows = []
    all_ap50 = []
    all_ap5095 = []

    for category_id, name in categories.items():
        class_predictions = [pred for pred in predictions if pred["category_id"] == category_id]
        class_ground_truths = {
            image_id: [ann["bbox"] for ann in anns if ann["category_id"] == category_id]
            for image_id, anns in ground_truths.items()
        }
        ap_by_threshold = [
            compute_ap_for_class(class_predictions, class_ground_truths, threshold)
            for threshold in thresholds
        ]
        valid_aps = [ap for ap in ap_by_threshold if ap is not None]
        ap50 = ap_by_threshold[0]
        ap5095 = sum(valid_aps) / len(valid_aps) if valid_aps else None

        if ap50 is not None:
            all_ap50.append(ap50)
        if ap5095 is not None:
            all_ap5095.append(ap5095)

        class_rows.append(
            {
                "name": name,
                "gt_count": sum(len(boxes) for boxes in class_ground_truths.values()),
                "prediction_count": len(class_predictions),
                "map50": ap50,
                "map5095": ap5095,
            }
        )

    return {
        "map50": sum(all_ap50) / len(all_ap50) if all_ap50 else None,
        "map5095": sum(all_ap5095) / len(all_ap5095) if all_ap5095 else None,
        "classes": class_rows,
    }


def format_score(value):
    return "-" if value is None else f"{value:.4f}"


def draw_annotations(image_path, output_path, ground_truths, predictions, categories):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for ann in ground_truths:
        x1, y1, x2, y2 = ann["bbox"]
        label = categories.get(ann["category_id"], str(ann["category_id"]))
        draw.rectangle([x1, y1, x2, y2], outline="#2563eb", width=3)
        draw.text((x1 + 3, max(0, y1 - 13)), f"GT {label}", fill="#2563eb", font=font)

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        label = categories.get(pred["category_id"], str(pred["category_id"]))
        draw.rectangle([x1, y1, x2, y2], outline="#f97316", width=3)
        draw.text(
            (x1 + 3, y1 + 3),
            f"P {label} {pred['score']:.2f}",
            fill="#f97316",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run_inference(model, image_paths, categories):
    mapper = category_mapper(model.names, categories)
    predictions = []
    predictions_by_image = defaultdict(list)

    results = model([str(path) for path in image_paths.values()], conf=0.25, verbose=False)
    path_to_image_id = {}
    for image_id, path in image_paths.items():
        path_to_image_id[str(path)] = image_id
        path_to_image_id[str(path.resolve())] = image_id

    for result in results:
        result_path = Path(result.path)
        image_id = path_to_image_id.get(str(result_path)) or path_to_image_id.get(str(result_path.resolve()))
        if image_id is None:
            continue

        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            model_class_id = int(box.cls[0])
            category_id = mapper(model_class_id)
            if category_id is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            prediction = {
                "image_id": image_id,
                "category_id": category_id,
                "score": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
            }
            predictions.append(prediction)
            predictions_by_image[image_id].append(prediction)

    return predictions, predictions_by_image


@app.template_filter("score")
def score_filter(value):
    return format_score(value)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            clean_previous_runs()
            run_id = uuid.uuid4().hex[:10]
            upload_dir = PRIVATE_RUNS_DIR / run_id / "uploads"
            result_dir = PUBLIC_RESULTS_DIR / run_id / "results"
            upload_dir.mkdir(parents=True, exist_ok=True)

            json_file = request.files.get("json_file")
            image_files = request.files.getlist("image_files")
            if not json_file or not json_file.filename:
                raise ValueError("Upload a COCO JSON file.")

            json_path = upload_dir / secure_filename(json_file.filename)
            json_file.save(json_path)

            coco_images, categories, annotations_by_image = parse_coco(json_path)
            saved_images = save_uploaded_images(image_files, upload_dir)
            resolved_images, missing_images = resolve_image_paths(coco_images, saved_images)
            if not resolved_images:
                raise ValueError("No uploaded images match file_name values in the COCO JSON.")

            model = load_yolo_model(request.files.get("model_file"))
            predictions, predictions_by_image = run_inference(model, resolved_images, categories)
            resolved_annotations_by_image = {
                image_id: annotations_by_image.get(image_id, []) for image_id in resolved_images
            }
            metrics = evaluate_map(predictions, resolved_annotations_by_image, categories)

            cases = []
            for image_id, image_path in resolved_images.items():
                image_info = coco_images[image_id]
                output_name = f"{image_id}_{secure_filename(Path(image_path).name)}"
                output_path = result_dir / output_name
                gt_items = resolved_annotations_by_image.get(image_id, [])
                pred_items = predictions_by_image.get(image_id, [])
                draw_annotations(image_path, output_path, gt_items, pred_items, categories)
                cases.append(
                    {
                        "file_name": image_info.get("file_name", Path(image_path).name),
                        "result_url": url_for("static", filename=f"runs/{run_id}/results/{output_name}"),
                        "gt_count": len(gt_items),
                        "prediction_count": len(pred_items),
                        "predictions": sorted(
                            [
                                {
                                    "label": categories.get(item["category_id"], str(item["category_id"])),
                                    "score": item["score"],
                                }
                                for item in pred_items
                            ],
                            key=lambda item: item["score"],
                            reverse=True,
                        ),
                    }
                )

            return render_template(
                "index.html",
                metrics=metrics,
                cases=cases,
                missing_images=missing_images,
                model_path=os.environ.get("MODEL_PATH"),
            )
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        metrics=None,
        cases=None,
        missing_images=None,
        model_path=os.environ.get("MODEL_PATH"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", "5000")))
