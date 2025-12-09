import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

df = pd.read_csv('annotation.csv')

# Distribution des classes
classes = np.unique(df['label'].values)
class_dist = df['label'].value_counts().sort_index()

# Graphique de la distribution des classes
fig, ax = plt.subplots()
ax.set_title('Distribution des classes', y = 1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.savefig('distribution_classes.png', dpi=150, bbox_inches='tight')
print("Graphique sauvegardé dans 'distribution_classes.png'")