#simple script to create bar grpah analysing results for report

import json
import matplotlib.pyplot as plt

#load metrics from JSON files
with open('cached_acc_n.json') as f:
    acc_baseline = json.load(f)['accuracy']
with open('cached_acc_ensemble.json') as f:
    acc_ensemble = json.load(f)['accuracy']
with open('cached_acc_n_hn.json') as f:
    acc_hn = json.load(f)['acc']
with open('cached_acc_ensemble_hn.json') as f:
    acc_ensemble_hn = json.load(f)['acc']

with open('cached_map_n.json') as f:
    map_baseline = json.load(f)['map50']
with open('cached_map_ensemble.json') as f:
    map_ensemble = json.load(f)['map50']
with open('cached_map_n_hn.json') as f:
    map_hn = json.load(f)['map50']
with open('cached_map_ensemble_hn.json') as f:
    map_ensemble_hn = json.load(f)['map50']

#prepare data
methods = ['Baseline', 'Ensemble', 'HNM', 'Ensemble+HNM']
accuracy = [acc_baseline, acc_ensemble, acc_hn, acc_ensemble_hn]
map50 = [map_baseline, map_ensemble, map_hn, map_ensemble_hn]

#plot accuracy
plt.figure()
plt.bar(methods, accuracy)
plt.ylabel('Test Accuracy')
plt.title('Comparison of Test Accuracy')
plt.ylim(0.7, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#plot mAP
plt.figure()
plt.bar(methods, map50)
plt.ylabel('mAP')
plt.title('Comparison of mAP')
plt.ylim(0.5, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
