# Structure du Projet Réorganisée

## 📁 Organisation des Dossiers

```
Quantnuis-1/
├── data/                      # Toutes les données
│   ├── raw/                   # Fichiers CSV bruts d'annotations
│   ├── slices/                # Segments audio d'entraînement
│   └── annotation.csv         # Fichier d'annotations principal
├── models/                    # Modèles entraînés et scalers
│   ├── model_improved.h5
│   ├── model_improved_best.h5
│   ├── model.h5
│   └── scaler.pkl
├── output/                    # Résultats et prédictions
│   ├── predictions_raw.csv
│   └── training_history_improved.png
└── Scripts Python à la racine
```

## 🚀 Utilisation Rapide

### Méthode Simple (Recommandée)

```bash
python main.py
```

Menu interactif avec toutes les options.

### Méthode Directe

```bash
# 1. Créer les annotations brutes
python annotation.py

# 2. Découper un fichier audio
python slicing.py [fichier_audio]

# 3. Entraîner le modèle
python model_improved.py

# 4. Prédire sur un fichier
python predict_improved.py [fichier_audio]

# 5. Analyser un fichier complet
python predict_full_audio.py [fichier_audio]
```

## 📝 Workflow Typique

1. **Créer les annotations** : `python annotation.py`
   - Génère `data/raw/annotations_raw.csv`

2. **Découper l'audio** : `python slicing.py mon_audio.wav`
   - Génère `data/slices/*.wav` et `data/annotation.csv`

3. **Entraîner** : `python model_improved.py`
   - Génère `models/model_improved.h5` et `models/scaler.pkl`

4. **Prédire** : `python predict_improved.py data/slices/slice_001.wav`

5. **Analyser un fichier complet** : `python predict_full_audio.py long_audio.wav`
   - Génère `output/predictions_raw.csv`
