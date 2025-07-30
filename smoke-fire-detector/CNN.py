#script for the baseline CNN model using YOLOv8n

import os
import glob
import json
import random

import cv2
from ultralytics import YOLO


#path to data.yaml file
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "smoke-fire", "data.yaml")
#where to store the best‐performing weights
CACHE_MODEL    = "cached_model.pt"

#where to store the evaluation metric
CACHE_METRICS  = "cached_metrics.json"

#folder for saving demo outputs
DEMO_OUTPUT    = "demo_output/baseline"

#function to retrieve best weights from runs folder
def find_latest_best():
    # match any runs/*/weights/best.pt
    candidates = glob.glob(os.path.join(BASE, "runs", "*", "weights", "best.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt found in runs/*/weights/")
    # return the most recently modified
    return max(candidates, key=os.path.getmtime)

#function to perform training on smoke/fire data using pretrained YOLOv8n model
def train_and_cache():
    #initialize model
    model = YOLO("yolov8n.pt")

    #train on smoke/fire data
    model.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        exist_ok=True,
        name="smoke_fire"
    )

    #locate the best weights and reload them using find_latest_best function
    best_weights = find_latest_best()
    model = YOLO(best_weights)

    #evaluate on the test set
    results = model.val(data=DATA_YAML, split="test")

    #retrieve mAP@0.5 score
    map50 = float(results.box.map50)
    metrics = {"map50": map50}

    #cache model and mAP score
    model.save(CACHE_MODEL)
    with open(CACHE_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Trained & cached model: {CACHE_MODEL}")
    print(f"   mAP@0.5 = {map50:.4f}")
    return model, metrics

#function to load cached model and metrics
def load_cache():
    model = YOLO(CACHE_MODEL)
    metrics = json.load(open(CACHE_METRICS))
    print(f"Loaded cached model: {CACHE_MODEL}")
    print(f"   mAP@0.5 = {metrics['map50']:.4f}")
    return model, metrics


#function to load cached model and metrics if they exist, or perform training if they dont
def load_or_train():
    #if caches exist call on load_cache function
    if os.path.exists(CACHE_MODEL) and os.path.exists(CACHE_METRICS):
        return load_cache()
    #if not, call on train_and_cache function
    else:
        return train_and_cache()

#function to choose a random image from test set and run a demo on it
def run_demo(model):
    os.makedirs(DEMO_OUTPUT, exist_ok=True)

    #pick a random test image
    candidates = glob.glob(os.path.join("smoke-fire","data", "test", "images", "*.*"))
    img_path = random.choice(candidates)
    print(f"Running demo on {img_path}")

    #run detection
    results = model.predict(img_path, conf=0.25, save=False)

    #draw the boxes & labels
    annotated = results[0].plot()

    #save resulting image
    out_path = os.path.join(DEMO_OUTPUT, os.path.basename(img_path))
    cv2.imwrite(out_path, annotated)
    print(f"Saved demo to {out_path}")

if __name__ == "__main__":
    #load cached data or train, then run a demo
    model, metrics = load_or_train()
    run_demo(model)
