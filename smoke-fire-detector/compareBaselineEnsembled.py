#script to create bar grpah comparing baseline and ensembled only models for use in presentation

import json
import matplotlib.pyplot as plt

# Load cached metrics
with open('cached_map_n.json') as f:
    map_base = json.load(f)['map50']
with open('cached_acc_n.json') as f:
    acc_base = json.load(f)['accuracy']
with open('cached_map_ensemble.json') as f:
    map_ens = json.load(f)['map50']
with open('cached_acc_ensemble.json') as f:
    acc_ens = json.load(f)['accuracy']

# Prepare data
metrics = ['mAP', 'Accuracy']
baseline_scores = [map_base, acc_base]
ensemble_scores = [map_ens, acc_ens]

x = range(len(metrics))
width = 0.35

# Plot
fig, ax = plt.subplots()
ax.bar([i - width/2 for i in x], baseline_scores, width, label='Baseline')
ax.bar([i + width/2 for i in x], ensemble_scores, width, label='Ensembled')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.legend()
ax.set_title('Baseline vs Ensemble Performance')

plt.tight_layout()
plt.show()