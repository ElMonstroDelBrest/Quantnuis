import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

# Configuration des chemins
DATA_DIR = "data"
ANNOTATION_CSV = os.path.join(DATA_DIR, "annotation.csv")
OUTPUT_DIR = "output"
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "distribution_classes.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(ANNOTATION_CSV)

# Distribution des classes
classes = np.unique(df['label'].values)
class_dist = df['label'].value_counts().sort_index()

# Graphique de la distribution des classes
fig, ax = plt.subplots()
ax.set_title('Distribution des classes', y = 1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"Graphique sauvegardé dans '{OUTPUT_IMG}'")