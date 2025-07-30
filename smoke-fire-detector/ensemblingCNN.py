#script for CNN with ensembling only

import os
import glob
import json
import random

import cv2
import numpy as np
from ultralytics import YOLO

#path for data.yaml
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "smoke-fire", "data.yaml")
#cached models and metrics without ensembling
CACHE_MODELS = ["cached_model_n.pt", "cached_model_m.pt", "cached_model_l.pt"]
CACHE_METRICS = ["cached_metrics_n.json", "cached_metrics_m.json", "cached_metrics_l.json"]
CACHE_IMAGE_ACC_N = "cached_acc_n.json"
#cached models and metrics for ensembling
CACHE_ENSEMBLE_METRICS = "cached_metrics_ensemble.json"
CACHE_IMAGE_ACC_ENSEMBLE = "cached_acc_ensemble.json"
DEMO_OUTPUT = "demo_output"
#paths for test and label directories
TEST_IMAGES = glob.glob(os.path.join(BASE, "smoke-fire", "data", "test", "images", "*.*"))
LABEL_DIR = os.path.join(BASE, "smoke-fire", "data", "test", "labels")

#function to find best weights for latest run
def find_latest_best(run_name):
    runs = glob.glob(os.path.join(BASE, "runs", run_name, "weights", "best.pt"))
    if not runs:
        raise FileNotFoundError(f"No best.pt for run {run_name}")
    return max(runs, key=os.path.getmtime)

#function to perform training on given model and cache
def train_model(variant, run_name, cache_model, cache_metrics):
    model = YOLO(variant)
    model.train(data=DATA_YAML, epochs=10, imgsz=640, batch=16,
                device=0, project="runs", name=run_name, exist_ok=True)
    best = find_latest_best(run_name)
    model = YOLO(best)
    results = model.val(data=DATA_YAML, split="test")
    map50 = float(results.box.map50)
    with open(cache_metrics, "w") as f:
        json.dump({"map50": map50}, f, indent=2)
    model.save(cache_model)
    print(f"{variant} cached as {cache_model}, mAP@0.5={map50:.4f}")
    return model

#function to load a cached model and metrics
def load_model(cache_model, cache_metrics):
    model = YOLO(cache_model)
    m = json.load(open(cache_metrics))
    print(f"Loaded {cache_model}, mAP@0.5={m['map50']:.4f}")
    return model, m

#function to hadnle whether a model can be loaded or needs to be trained
def load_or_train():
    models, mets = [], []
    configs = [
        ("yolov8n.pt", "n", CACHE_MODELS[0], CACHE_METRICS[0]),
        ("yolov8m.pt", "m", CACHE_MODELS[1], CACHE_METRICS[1]),
        ("yolov8l.pt", "l", CACHE_MODELS[2], CACHE_METRICS[2]),
    ]
    for variant, tag, mfile, jfile in configs:
        if os.path.exists(mfile) and os.path.exists(jfile):
            mdl, met = load_model(mfile, jfile)
        else:
            mdl = train_model(variant, f"smoke_fire_{tag}", mfile, jfile)
            met = json.load(open(jfile))
        models.append(mdl)
        mets.append(met)
    return models, mets

#function to perform wighted box fusion for ensembling
def weighted_box_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5):
    #stack all boxes, scores, and labels
    boxes = np.vstack(boxes_list)
    scores = np.hstack(scores_list)
    labels = np.hstack(labels_list)
    #process both classes separately
    fused_boxes, fused_scores, fused_labels = [], [], []
    for cls in np.unique(labels):
        inds = np.where(labels == cls)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        used = np.zeros(len(inds), bool)
        #cluster by highest score
        for i in np.argsort(-cls_scores):
            if used[i]:
                continue
            same = [i]
            used[i] = True
            #find overlapping boxes
            for j in range(len(inds)):
                if not used[j] and compute_iou(cls_boxes[i], cls_boxes[j]) > iou_thr:
                    same.append(j)
                    used[j] = True
            cluster = cls_boxes[same]
            confs = cls_scores[same]
            #weight by confidence
            w = confs / confs.sum()
            #compute weighted box averages
            fb = np.dot(w, cluster)
            fused_boxes.append(fb)
            fused_scores.append(confs.max())
            fused_labels.append(cls)
    if not fused_boxes:
        return np.zeros((0, 6))
    out = np.hstack((np.array(fused_boxes), np.array(fused_scores)[:, None], np.array(fused_labels)[:, None]))
    return out

#function to run on an image with each model, collect boxes, scores, and labels, then fuse using weighted box fusion
def ensemble_predictions(models, img_path, conf_thres=0.25, iou_thr=0.5):
    boxes_list, scores_list, labels_list = [], [], []
    for m in models:
        res = m.predict(img_path, conf=conf_thres, save=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        b = res.boxes.xyxy.cpu().numpy()
        s = res.boxes.conf.cpu().numpy()
        c = res.boxes.cls.cpu().numpy()
        boxes_list.append(b)
        scores_list.append(s)
        labels_list.append(c)
    if not boxes_list:
        return np.zeros((0, 6))
    return weighted_box_fusion(boxes_list, scores_list, labels_list, iou_thr)

#function to compute intersection over union for two bounding boxes for ensembling
def compute_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

##function to compute average precision (AP) to be averaged for mAP across recall levels
def compute_average_precision(rec, prec):
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
        ap += p / 11
    return ap

#function to compute mAP for fused detectors across images + their ground truth labels
def evaluate_ensemble_map(models):
    #load ground truth labels per image
    gt, dets = {}, {0: [], 1: []}
    for img in TEST_IMAGES:
        h, w = cv2.imread(img).shape[:2]
        lbl = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img))[0] + '.txt')
        boxes = []
        if os.path.exists(lbl):
            for ln in open(lbl):
                cl, x, y, bw, bh = map(float, ln.split())
                cx, cy = x * w, y * h
                bw, bh = bw * w, bh * h
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                boxes.append((int(cl), [x1, y1, x2, y2]))
        gt[img] = boxes
    #collect all fused detections for both calsses
    for img in TEST_IMAGES:
        for x1, y1, x2, y2, conf, cl in ensemble_predictions(models, img):
            dets[int(cl)].append((img, conf, [x1, y1, x2, y2]))
    #compute AP for both classes
    ap_list = []
    for cl in [0, 1]:
        #sort by descending confidence
        items = sorted(dets[cl], key=lambda x: -x[1])
        flags = {img: [False] * len([b for c, b in gt[img] if c == cl]) for img in TEST_IMAGES}
        tp, fp = [], []
        total = sum(1 for img in TEST_IMAGES for c, b in gt[img] if c == cl)
        for img, conf, box in items:
            ious = [compute_iou(box, b) for c, b in gt[img] if c == cl]
            if ious and max(ious) >= 0.5:
                idx = ious.index(max(ious))
                if not flags[img][idx]:
                    tp.append(1); fp.append(0); flags[img][idx] = True
                else:
                    tp.append(0); fp.append(1)
            else:
                tp.append(0); fp.append(1)
        rec = np.cumsum(tp) / (total + 1e-6)
        prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp) + 1e-6)
        ap_list.append(compute_average_precision(rec, prec))
    #return the average
    m50 = np.mean(ap_list)
    print(f"Ensemble mAP@0.5 on test set: {m50:.4f}")
    return m50

#function to run a demo on random test image
def run_demo(models):
    os.makedirs(DEMO_OUTPUT, exist_ok=True)
    img = random.choice(TEST_IMAGES)
    img_mat = cv2.imread(img)
    print(f"Demo on {img}")
    for x1, y1, x2, y2, conf, cl in ensemble_predictions(models, img):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        col = (0, 0, 255) if cl == 1 else (255, 0, 0)
        cv2.rectangle(img_mat, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img_mat, f"{models[0].names[cl]} {conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    op = os.path.join(DEMO_OUTPUT, os.path.basename(img))
    cv2.imwrite(op, img_mat)
    print(f"Saved demo to {op}")

#function to compute simple accuracy (only accounts for correct classifications)
def evaluate_image_accuracy(models, name="Ensembled"):
    correct = 0
    for img_path in TEST_IMAGES:
        gt_labels = set()
        lbl = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if os.path.exists(lbl):
            for ln in open(lbl):
                gt_labels.add(int(float(ln.split()[0])))
        dets = ensemble_predictions(models, img_path)
        pred_labels = set(dets[:,5].astype(int)) if dets.shape[0]>0 else set()
        if pred_labels == gt_labels:
            correct += 1
    acc = correct / len(TEST_IMAGES)
    print(f"{name} model image-level accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    #load or train models and metrics
    models, mets = load_or_train()
    #display mAP for all models
    names = ["YOLOv8n", "YOLOv8m", "YOLOv8l"]
    for name, met in zip(names, mets):
        print(f"{name} mAP@0.5: {met['map50']:.4f}")

    #display YOLOv8n (baseline model) simple accuracy (cache)
    if os.path.exists(CACHE_IMAGE_ACC_N):
        acc_n = json.load(open(CACHE_IMAGE_ACC_N))["accuracy"]
        print(f"YOLOv8n cached image-level accuracy: {acc_n:.4f}")
    else:
        acc_n = evaluate_image_accuracy([models[0]], name="YOLOv8n")
        with open(CACHE_IMAGE_ACC_N, "w") as f:
            json.dump({"accuracy": acc_n}, f, indent=2)

    #display ensemble mAP (cache)
    if os.path.exists(CACHE_ENSEMBLE_METRICS):
        ensemble_map50 = json.load(open(CACHE_ENSEMBLE_METRICS))["map50"]
        print(f"Cached ensemble mAP@0.5: {ensemble_map50:.4f}")
    else:
        ensemble_map50 = evaluate_ensemble_map(models)
        with open(CACHE_ENSEMBLE_METRICS, "w") as f:
            json.dump({"map50": ensemble_map50}, f, indent=2)

    #dispaly ensemble simple accuracy accuracy (cache)
    if os.path.exists(CACHE_IMAGE_ACC_ENSEMBLE):
        acc_ensemble = json.load(open(CACHE_IMAGE_ACC_ENSEMBLE))["accuracy"]
        print(f"Cached ensemble image-level accuracy: {acc_ensemble:.4f}")
    else:
        acc_ensemble = evaluate_image_accuracy(models, name="Ensembled")
        with open(CACHE_IMAGE_ACC_ENSEMBLE, "w") as f:
            json.dump({"accuracy": acc_ensemble}, f, indent=2)

    #run a demo
    run_demo(models)
