#script for the CNN model with ensembling and hard negative mining

import os
import glob
import json
import random
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

#path for data.yaml
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "smoke-fire", "data.yaml")

#caches
CACHE_MODELS = ["cached_model_n.pt", "cached_model_m.pt", "cached_model_l.pt"]
CACHE_METRICS = ["cached_map_n.json", "cached_map_m.json", "cached_map_l.json"]
CACHE_ENSEMBLE_MAP = "cached_map_ensemble.json"
CACHE_ACC_N = "cached_acc_n.json"
CACHE_ACC_ENSEMBLE = "cached_acc_ensemble.json"

#post-hard negative mining caches (hard negative mining only as well as hard negative mining + ensembling)
CACHE_MODELS_HN = ["cached_model_n_hn.pt",
                   "cached_model_m_hn.pt",
                   "cached_model_l_hn.pt"]
CACHE_METRICS_HN = ["cached_map_n_hn.json", "cached_map_m_hn.json", "cached_map_l_hn.json"]
CACHE_ENSEMBLE_MAP_HN = "cached_map_ensemble_hn.json"
CACHE_ACC_N_HN = "cached_acc_n_hn.json"
CACHE_ACC_ENSEMBLE_HN = "cached_acc_ensemble_hn.json"
CACHE_HARD_NEG = "cached_hard_neg.json"

#output folder for demos
DEMO_OUTPUT = "demo_output"
#paths for image and label directories
TRAIN_IMG_DIR = os.path.join(BASE, "smoke-fire", "data", "train", "images")
TRAIN_LABEL_DIR = os.path.join(BASE, "smoke-fire", "data", "train", "labels")
TEST_IMAGES = glob.glob(os.path.join(BASE, "smoke-fire", "data", "test", "images", "*.*"))
LABEL_DIR = os.path.join(BASE, "smoke-fire", "data", "test", "labels")

#function to retrieve best weights from runs folder
def find_latest_best(run_name):
    runs = glob.glob(os.path.join(BASE, "runs", run_name, "weights", "best.pt"))
    if not runs:
        raise FileNotFoundError(f"No best.pt for run {run_name}")
    return max(runs, key=os.path.getmtime)

#function to compute intersection over union for two bounding boxes for ensembling
def compute_iou(b1, b2):
    #intersection coordinmates
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    #intersection area
    inter = max(0, xB-xA) * max(0, yB-yA)
    #areas of each box
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    #return intersection over union
    return inter / (a1 + a2 - inter + 1e-6)

#function to compute average precision (AP) to be averaged for mAP across recall levels
#rec = array of recall values
#prec = array of precision values
def compute_average_precision(rec, prec):
    ap = 0.0
    for t in np.arange(0,1.1,0.1):
        #precison at recall >= t
        p = np.max(prec[rec>=t]) if np.sum(rec>=t)>0 else 0
        ap += p / 11.0
    return ap

#function to train given model variant and cache
def train_model(variant, run_name, cache_model, cache_metric):
    #train given model
    model = YOLO(variant)
    model.train(data=DATA_YAML, epochs=10, imgsz=640, batch=16,
                device=0, project="runs", name=run_name, exist_ok=True)
    #finf best weights
    best = find_latest_best(run_name)
    model = YOLO(best)
    #validate on test set
    results = model.val(data=DATA_YAML, split="test")
    m50 = float(results.box.map50)
    #cache
    model.save(cache_model)
    with open(cache_metric, 'w') as f:
        json.dump({'map50': m50}, f, indent=2)
    print(f"{variant} -> {cache_model}, mAP@0.5={m50:.4f}")
    return model

#function to load cached models or perform training, testing, and caching if needed
def load_or_train_models(model_caches, metric_caches):
    models = []
    for variant, tag, mc, met in zip(
        ["yolov8n.pt","yolov8m.pt","yolov8l.pt"],
        ['n','m','l'],
        model_caches,
        metric_caches
    ):
        if os.path.exists(mc) and os.path.exists(met):
            print(f"Loaded cached {mc}")
            models.append(YOLO(mc))
        else:
            models.append(train_model(variant, f"smoke_fire_{tag}", mc, met))
    return models

#function to perform weighted box fusion to fuse results from each model (ensembling)
def weighted_box_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5):
    #stack all detections
    boxes = np.vstack(boxes_list)
    scores = np.hstack(scores_list)
    labels = np.hstack(labels_list)
    fused = []
    #process both calsses separetely 
    for cls in np.unique(labels):
        idxs = np.where(labels==cls)[0]
        bxs, scs = boxes[idxs], scores[idxs]
        used = np.zeros(len(idxs), bool)
        #cluster by highest score
        for i in np.argsort(-scs):
            if used[i]: continue
            grp=[i]; used[i]=True
            #find overlapping boxes
            for j in range(len(idxs)):
                if not used[j] and compute_iou(bxs[i], bxs[j]) > iou_thr:
                    grp.append(j); used[j]=True
            #weight by confidence
            ws = scs[grp] / scs[grp].sum()
            #compute weighted box averages
            fb = (ws[:,None]*bxs[grp]).sum(axis=0)
            fused.append([*fb, scs[grp].max(), cls])
    return np.array(fused) if fused else np.zeros((0,6))

#function to run on an image with each model, collect boxes, scores, and labels, then fuse using weighted box fusion
def ensemble_predictions(models, img, conf_thres=0.25, iou_thr=0.5):
    bl, sl, ll = [], [], []
    for m in models:
        res = m.predict(img, conf=conf_thres, save=False)[0]
        if res.boxes:
            bl.append(res.boxes.xyxy.cpu().numpy())
            sl.append(res.boxes.conf.cpu().numpy())
            ll.append(res.boxes.cls.cpu().numpy())
    return weighted_box_fusion(bl, sl, ll, iou_thr) if bl else np.zeros((0,6))

#function to compute mAP for fused detectors across images + their ground truth labels
def evaluate_ensemble_map(models, images, label_dir):
    #load ground truth labels per image
    gt, dets = {}, {0:[],1:[]}
    for img in images:
        h,w = cv2.imread(img).shape[:2]
        lbl = os.path.join(label_dir, os.path.splitext(os.path.basename(img))[0]+'.txt')
        boxes=[]
        if os.path.exists(lbl):
            for ln in open(lbl):
                c,x,y,bw,bh = map(float,ln.split())
                cx,cy = x*w, y*h
                bw, bh = bw*w, bh*h
                x1,y1 = cx-bw/2, cy-bh/2
                x2,y2 = cx+bw/2, cy+bh/2
                boxes.append((int(c),[x1,y1,x2,y2]))
        gt[img] = boxes
    #collect all fused detections for both calsses
    for img in images:
        for x1,y1,x2,y2,conf,cl in ensemble_predictions(models,img):
            dets[int(cl)].append((img,conf,[x1,y1,x2,y2]))
    #compute AP for both classes
    ap_list=[]
    for cl in [0,1]:
        #sort by descending confidence
        items = sorted(dets[cl], key=lambda x:-x[1])
        flags = {img:[False]*len([b for c,b in gt[img] if c==cl]) for img in images}
        tp,fp = [],[]
        total = sum(1 for img in images for c,b in gt[img] if c==cl)
        for img,conf,box in items:
            ious = [compute_iou(box,b) for c,b in gt[img] if c==cl]
            if ious and max(ious)>=0.5 and not flags[img][np.argmax(ious)]:
                tp.append(1); fp.append(0);
                flags[img][np.argmax(ious)] = True
            else:
                tp.append(0); fp.append(1)
        rec = np.cumsum(tp)/(total+1e-6)
        prec = np.cumsum(tp)/(np.cumsum(tp)+np.cumsum(fp)+1e-6)
        ap_list.append(compute_average_precision(rec,prec))
    #return the average
    return np.mean(ap_list)

#function to calculate a simple accuracy score (predicted classes in image = ground truth)
def evaluate_image_accuracy(models, images, label_dir):
    correct=0
    for img in images:
        gt_labels=set()
        lbl = os.path.join(label_dir, os.path.splitext(os.path.basename(img))[0]+'.txt')
        if os.path.exists(lbl):
            for ln in open(lbl):
                gt_labels.add(int(float(ln.split()[0])))
        dets = ensemble_predictions(models,img)
        pred_labels = set(dets[:,5].astype(int)) if dets.size>0 else set()
        if pred_labels == gt_labels:
            correct+=1
    return correct/len(images)

#function to perform hard negative mining
#identify false positives, add them to training set, return list of images added
def mine_hard_negatives(models, conf_thres=0.5, iou_thr=0.5):
    #all the label files in the training set
    negs = glob.glob(os.path.join(TRAIN_LABEL_DIR, "*.txt"))
    #directory for storing hard negatives found for reference
    hard_dir = os.path.join(BASE, "hard_negatives")
    os.makedirs(hard_dir, exist_ok=True)
    mined=[]
    for lbl in negs:
        fname=os.path.basename(lbl)
        #skip if current image was already found as a hard negative previously
        if fname.startswith("HNM_") or os.path.getsize(lbl)>0:
            continue
        base=os.path.splitext(fname)[0]
        imgs=glob.glob(os.path.join(TRAIN_IMG_DIR,base+".*"))
        if not imgs: continue
        #evaluate models on image
        dets = ensemble_predictions(models,imgs[0],conf_thres,iou_thr)
        #if detections found on image with no smoke or fire, it is a hard negative
        if dets.shape[0]>0:
            #add to training set with HNM prefix to distinguish them from original images
            ext=os.path.splitext(imgs[0])[1]
            newn=f"HNM_{base}{ext}"
            shutil.copy(imgs[0],os.path.join(TRAIN_IMG_DIR,newn))
            open(os.path.join(TRAIN_LABEL_DIR,f"{os.path.splitext(newn)[0]}.txt"),'w').close()
            shutil.copy(imgs[0],os.path.join(hard_dir,newn))
            mined.append(newn)
    print(f"{len(mined)} hard negatives mined and added to training set")
    return mined

if __name__ == '__main__':
    #load baseline and ensemble only metrics if they exist
    print("--- Baseline and Ensembling Only Metrics ---")
    baseline_models = load_or_train_models(CACHE_MODELS,    CACHE_METRICS)
    for name, cm in zip(["YOLOv8n","YOLOv8m","YOLOv8l"], CACHE_METRICS):
        print(f"{name} baseline mAP@0.5: {json.load(open(cm))['map50']:.4f}")
    if os.path.exists(CACHE_ENSEMBLE_MAP):
        em = json.load(open(CACHE_ENSEMBLE_MAP))['map50']
    else:
        em = evaluate_ensemble_map(baseline_models, TEST_IMAGES, LABEL_DIR)
        with open(CACHE_ENSEMBLE_MAP,'w') as f: json.dump({'map50':em},f,indent=2)
    print(f"Ensemble only mAP@0.5: {em:.4f}")
    if os.path.exists(CACHE_ACC_N):
        accn = json.load(open(CACHE_ACC_N))['accuracy']
    else:
        accn = evaluate_image_accuracy([baseline_models[0]], TEST_IMAGES, LABEL_DIR)
        with open(CACHE_ACC_N,'w') as f: json.dump({'accuracy':accn},f,indent=2)
    print(f"YOLOv8n baseline accuracy: {accn:.4f}")
    if os.path.exists(CACHE_ACC_ENSEMBLE):
        ace = json.load(open(CACHE_ACC_ENSEMBLE))['accuracy']
    else:
        ace = evaluate_image_accuracy(baseline_models, TEST_IMAGES, LABEL_DIR)
        with open(CACHE_ACC_ENSEMBLE,'w') as f: json.dump({'accuracy':ace},f,indent=2)
    print(f"Ensemble only accuracy: {ace:.4f}")

    #perform hard negative mining
    print("--- Hard Negative Mining ---")
    if os.path.exists(CACHE_HARD_NEG):
        mined = json.load(open(CACHE_HARD_NEG))['mined']
        print(f"Reusing {len(mined)} cached hard negatives")
    else:
        mined = mine_hard_negatives(baseline_models)
        with open(CACHE_HARD_NEG,'w') as f: json.dump({'mined':mined},f,indent=2)

    print("--- Post-Hard-Negative Metrics ---")
    #load or retrain HNM models for mAP caches
    if all(os.path.exists(c) for c in CACHE_METRICS_HN):
        for name, cm in zip(["YOLOv8n", "YOLOv8m", "YOLOv8l"], CACHE_METRICS_HN):
            print(f"{name} HN mAP@0.5: {json.load(open(cm))['map50']:.4f}")
    else:
        hn_models = load_or_train_models(CACHE_MODELS_HN, CACHE_METRICS_HN)
        for name, cm in zip(["YOLOv8n", "YOLOv8m", "YOLOv8l"], CACHE_METRICS_HN):
            print(f"{name} HN mAP@0.5: {json.load(open(cm))['map50']:.4f}")
    #ensemble + HNM mAP
    if os.path.exists(CACHE_ENSEMBLE_MAP_HN):
        em_hn = json.load(open(CACHE_ENSEMBLE_MAP_HN))['map50']
    else:
        em_hn = evaluate_ensemble_map(hn_models, TEST_IMAGES, LABEL_DIR)
        with open(CACHE_ENSEMBLE_MAP_HN, 'w') as f: json.dump({'map50': em_hn}, f, indent=2)
    print(f"Ensemble HN mAP@0.5: {em_hn:.4f}")
    #YOLOv8n HNM accuracy (baseline model + HNM, no ensembling)
    if os.path.exists(CACHE_ACC_N_HN):
        accn_hn = json.load(open(CACHE_ACC_N_HN))['acc']
    else:
        accn_hn = evaluate_image_accuracy([hn_models[0]], TEST_IMAGES, LABEL_DIR)
        with open(CACHE_ACC_N_HN, 'w') as f: json.dump({'acc': accn_hn}, f, indent=2)
    print(f"YOLOv8n HNM accuracy: {accn_hn:.4f}")
    # Ensemble HN accuracy
    if os.path.exists(CACHE_ACC_ENSEMBLE_HN):
        ace_hn = json.load(open(CACHE_ACC_ENSEMBLE_HN))['acc']
    else:
        ace_hn = evaluate_image_accuracy(hn_models, TEST_IMAGES, LABEL_DIR)
        with open(CACHE_ACC_ENSEMBLE_HN, 'w') as f: json.dump({'acc': ace_hn}, f, indent=2)
    print(f"Ensemble + HNM accuracy: {ace_hn:.4f}")

    #run a demo
    os.makedirs(DEMO_OUTPUT, exist_ok=True)
    img = random.choice(TEST_IMAGES)
    print(f"Demo on {img}")
    mat = cv2.imread(img)
    for x1,y1,x2,y2,conf,cl in ensemble_predictions(baseline_models,img):
        col = (0,0,255) if cl==1 else (255,0,0)
        cv2.rectangle(mat,(int(x1),int(y1)),(int(x2),int(y2)),col,2)
        cv2.putText(mat,f"{baseline_models[0].names[int(cl)]} {conf:.2f}",
                    (int(x1),int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
    out = os.path.join(DEMO_OUTPUT, os.path.basename(img))
    cv2.imwrite(out, mat)
    print(f"Saved demo to {out}")
