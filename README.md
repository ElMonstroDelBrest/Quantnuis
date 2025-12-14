# 🎵 Système de Classification Audio avec TensorFlow

Ce projet implémente un système de classification audio utilisant des réseaux de neurones profonds pour identifier différentes classes de sons dans des fichiers audio.

## 📋 Table des Matières

- [Description](#description)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Ajout de Nouvelles Données](#ajout-de-nouvelles-données)
- [Paramètres du Modèle](#paramètres-du-modèle)
- [Format des Données](#format-des-données)
- [Scripts Disponibles](#scripts-disponibles)
- [Dépannage](#dépannage)

---

## 📖 Description

Ce système permet de :
- **Entraîner** un modèle de classification audio sur des segments audio étiquetés
- **Prédire** la classe d'un fichier audio ou d'un segment
- **Analyser** des fichiers audio complets (plusieurs heures) en les segmentant automatiquement
- **Visualiser** les résultats sous forme de graphiques et de fichiers CSV

Le modèle utilise des caractéristiques audio avancées (mel-spectrogramme, MFCC, chroma, etc.) et une architecture de réseau de neurones avec régularisation pour éviter le surapprentissage.

---

## 🔧 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des Dépendances

```bash
# Créer un environnement virtuel (recommandé)
python3 -m venv venv

# Activer l'environnement virtuel
# Sur Linux/Mac:
source venv/bin/activate
# Sur Windows:
# venv\Scripts\activate

# Installer les dépendances de base
pip install librosa soundfile pandas numpy scikit-learn matplotlib

# Installer TensorFlow (choisir selon votre configuration)
# Pour CPU uniquement:
pip install tensorflow

# Pour GPU NVIDIA (recommandé si vous avez un GPU):
pip install tensorflow[and-cuda]

# Vérifier la configuration GPU:
python test_gpu.py
```

> **📖 Guide GPU :** Consultez `SETUP_GPU.md` pour la configuration détaillée du GPU NVIDIA.

### Dépendances Principales

- **tensorflow** : Framework de deep learning
- **librosa** : Traitement et analyse audio
- **soundfile** : Lecture/écriture de fichiers audio
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **scikit-learn** : Outils de machine learning
- **matplotlib** : Visualisation

---

## 📁 Structure du Projet

```
Quantnuis-1/
├── data/                      # Dossier contenant toutes les données
│   ├── raw/                   # Fichiers CSV bruts d'annotations
│   │   └── annotations_raw.csv
│   ├── slices/                # Segments audio d'entraînement
│   │   ├── slice_001.wav
│   │   ├── slice_002.wav
│   │   └── ...
│   └── annotation.csv         # Fichier d'annotations principal (labels des segments)
├── models/                    # Modèles entraînés et scalers
│   ├── model_improved.h5      # Modèle entraîné sauvegardé
│   ├── model_improved_best.h5 # Meilleur modèle (selon validation)
│   └── scaler.pkl             # Scaler pour normaliser les nouvelles données
├── output/                    # Résultats et prédictions
│   ├── predictions_raw.csv    # Prédictions générées
│   └── training_history_improved.png
├── main.py                    # Script principal (menu interactif)
├── annotation.py              # Script pour créer le fichier d'annotations
├── slicing.py                 # Script pour découper un fichier audio en segments
├── model_improved.py          # Script d'entraînement du modèle amélioré
├── predict_improved.py        # Script de prédiction pour un fichier
├── predict_full_audio.py      # Script d'analyse d'un fichier audio complet
└── README.md                  # Ce fichier
```

---

## 🚀 Utilisation

### Méthode Simple : Script Principal

Le moyen le plus simple de lancer les différentes étapes est d'utiliser le script principal :

```bash
python main.py
```

Cela affiche un menu interactif avec toutes les options disponibles.

Vous pouvez aussi lancer directement une action :

```bash
python main.py 1  # Créer les annotations
python main.py 2  # Découper un fichier audio
python main.py 3  # Entraîner le modèle
python main.py 4  # Prédire sur un fichier
python main.py 5  # Analyser un fichier complet
```

### Méthode Avancée : Scripts Individuels

#### 1. Créer le Fichier d'Annotations

```bash
python annotation.py
```

Génère `data/raw/annotations_raw.csv` à partir des données brutes dans le script.

#### 2. Découper un Fichier Audio

```bash
python slicing.py [chemin_vers_fichier_audio]
```

**Ce que fait le script :**
- Lit les annotations depuis `data/raw/annotations_raw.csv` (ou le plus récent fichier `*_annotations_raw.csv`)
- Découpe le fichier audio en segments selon les timestamps
- Sauvegarde les segments dans `data/slices/`
- Crée `data/annotation.csv` avec les métadonnées

**Sans argument**, le script cherche automatiquement un fichier audio dans le répertoire courant.

#### 3. Entraîner le Modèle

```bash
python model_improved.py
```

**Ce que fait le script :**
- Extrait les caractéristiques audio de tous les fichiers dans `data/slices/`
- Normalise les données
- Augmente les données (ajout de bruit, time stretching, pitch shifting)
- Entraîne le modèle avec validation
- Sauvegarde le modèle dans `models/model_improved.h5`
- Sauvegarde le scaler dans `models/scaler.pkl`
- Génère des graphiques dans `output/training_history_improved.png`

**Fichiers générés :**
- `models/model_improved.h5` : Modèle final
- `models/model_improved_best.h5` : Meilleur modèle (selon validation loss)
- `models/scaler.pkl` : Normaliseur pour les nouvelles prédictions
- `output/training_history_improved.png` : Graphiques d'évolution

#### 4. Faire une Prédiction sur un Fichier

```bash
python predict_improved.py [chemin_vers_fichier_audio]
```

**Exemple :**
```bash
python predict_improved.py data/slices/slice_001.wav
```

**Sans argument**, le script utilise le premier fichier `.wav` trouvé dans `data/slices/`.

#### 5. Analyser un Fichier Audio Complet

```bash
python predict_full_audio.py [chemin_vers_fichier_audio]
```

**Paramètres configurables** (dans le script) :
- `segment_duration` : Durée de chaque segment en secondes (défaut: 30)
- `overlap` : Chevauchement entre segments en secondes (défaut: 5)
- `min_confidence` : Confiance minimale pour inclure une prédiction (défaut: 50%)

**Exemple :**
```bash
python predict_full_audio.py mon_fichier_audio.wav
```

Le script génère un fichier CSV (`output/predictions_raw.csv`) avec les prédictions pour chaque segment.

---

## 📥 Ajout de Nouvelles Données

> **📖 Guide détaillé :** Consultez `AJOUT_DONNEES.md` pour un guide complet étape par étape.

### Workflow Rapide

1. **Préparer le CSV d'annotations** : Placez votre fichier CSV dans `data/raw/` avec le format :
   ```csv
   Start,End,Label,Reliability
   00:09:34,00:10:12,1,3
   ```

2. **Découper l'audio** : 
   ```bash
   python main.py 2
   # Entrez le chemin vers votre fichier audio
   ```
   Les nouveaux slices seront **automatiquement ajoutés** à `data/slices/` et les annotations à `data/annotation.csv` (sans doublons).

3. **Réentraîner le modèle** (optionnel) :
   ```bash
   python main.py 3
   ```

### Méthode 1 : Ajout Manuel de Segments

#### Étape 1 : Créer les Annotations Brutes

1. **Créer le fichier d'annotations brutes** :
   ```bash
   python annotation.py
   ```
   
   Cela génère `data/raw/annotations_raw.csv` avec les timestamps et labels.

#### Étape 2 : Découper le Fichier Audio

1. **Découper votre fichier audio en segments** :
   ```bash
   python slicing.py [fichier_audio]
   ```

   Le script `slicing.py` :
   - Lit automatiquement `data/raw/annotations_raw.csv` (ou le plus récent)
   - Découpe le fichier audio en segments basés sur les annotations
   - Sauvegarde les segments dans `data/slices/`
   - Crée `data/annotation.csv` avec les métadonnées

#### Étape 3 : Vérifier le Fichier d'Annotations

Le fichier `data/annotation.csv` doit avoir le format suivant :

```csv
nfile,length,label,reliability
slice_001.wav,38,1,3
slice_002.wav,13,2,3
slice_028.wav,25,1,3
slice_029.wav,30,2,2
```

**Colonnes :**
- `nfile` : Nom du fichier (doit correspondre au fichier dans `slices/`)
- `length` : Durée du segment en secondes
- `label` : Classe/label du segment (entier, ex: 1, 2, 3, 4)
- `reliability` : Fiabilité de l'annotation (1-3, où 3 = très fiable)

**Note :** Le fichier `data/annotation.csv` est généré automatiquement par `slicing.py`. Si vous ajoutez manuellement des fichiers audio dans `data/slices/`, vous devrez mettre à jour `data/annotation.csv` manuellement.

#### Étape 4 : Réentraîner le Modèle

Une fois les nouvelles données ajoutées :

```bash
python model_improved.py
```

Le modèle sera réentraîné avec toutes les données (anciennes + nouvelles).

### Méthode 2 : Utiliser le Script d'Annotation

Le script `annotation.py` crée le fichier `data/raw/annotations_raw.csv` à partir de données brutes. Modifiez la variable `raw_data` dans le script pour ajouter vos propres annotations.

---

## ⚙️ Paramètres du Modèle

### Extraction de Caractéristiques

Le modèle extrait **304 caractéristiques** par fichier audio :

| Caractéristique | Nombre | Description |
|----------------|--------|-------------|
| **Mel-spectrogramme** | 256 | Moyenne (128) + Écart-type (128) |
| **MFCC** | 26 | 13 coefficients avec moyenne et écart-type |
| **Chroma** | 12 | Caractéristiques harmoniques |
| **Spectral Contrast** | 7 | Contraste spectral |
| **Zero Crossing Rate** | 2 | Moyenne et écart-type |
| **Tempo** | 1 | Estimation du tempo (BPM) |
| **TOTAL** | **304** | |

**Paramètres d'extraction :**
- Sample rate : 22050 Hz
- Mel bands : 128
- MFCC coefficients : 13

### Architecture du Modèle

```
Input (304 features)
  ↓
Dense(512) + BatchNormalization + Dropout(0.5)
  ↓
Dense(256) + BatchNormalization + Dropout(0.4)
  ↓
Dense(128) + BatchNormalization + Dropout(0.3)
  ↓
Dense(64) + BatchNormalization + Dropout(0.2)
  ↓
Output (num_classes) + Softmax
```

**Hyperparamètres :**
- **Optimiseur** : Adam
- **Learning rate** : 0.0001 (avec decay)
- **Loss function** : Categorical crossentropy
- **Batch size** : 16
- **Epochs** : 200 (avec early stopping)
- **Early stopping patience** : 20 epochs
- **Reduce LR patience** : 10 epochs

### Augmentation de Données

Pour chaque échantillon d'entraînement, le modèle crée plusieurs versions augmentées :
- **Bruit gaussien** : Ajout de bruit léger (σ=0.005)
- **Time stretching** : Ralentissement (rate=0.9) et accélération (rate=1.1)
- **Pitch shifting** : Modification de la hauteur tonale (±2 demi-tons)

### Normalisation

- **StandardScaler** : Normalisation z-score (moyenne=0, écart-type=1)
- Le scaler est sauvegardé dans `scaler.pkl` pour être réutilisé lors des prédictions

---

## 📊 Format des Données

### Fichier d'Annotations Brutes (`data/raw/annotations_raw.csv`)

Format des annotations brutes avec timestamps :

```csv
Start,End,Label,Reliability
00:09:34,00:10:12,1,3
00:11:30,00:11:43,2,3
```

### Fichier d'Annotations Principal (`data/annotation.csv`)

```csv
nfile,length,label,reliability
slice_001.wav,38,1,3
slice_002.wav,13,2,3
```

**Labels :**
- Les labels sont des entiers (1, 2, 3, 4, etc.)
- Le nombre de classes est déterminé automatiquement
- **Important** : Chaque classe doit avoir au moins 2 échantillons pour permettre la stratification

**Reliability :**
- 1 : Faible fiabilité
- 2 : Fiabilité moyenne
- 3 : Haute fiabilité

### Fichier Audio

**Format supporté :**
- `.wav` (recommandé)
- `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac` (via librosa)

**Recommandations :**
- Durée des segments : 10-60 secondes (optimal : 20-40 secondes)
- Sample rate : Toute valeur (sera convertie à 22050 Hz)
- Canaux : Mono ou stéréo (sera converti en mono)

### Fichier de Prédictions (`output/predictions_raw.csv`)

Format généré par `predict_full_audio.py` :

```csv
Start,End,Label,Reliability
00:03:20,00:03:50,4,1
00:09:35,00:10:05,4,1
```

**Colonnes :**
- `Start` : Timestamp de début (HH:MM:SS)
- `End` : Timestamp de fin (HH:MM:SS)
- `Label` : Classe prédite
- `Reliability` : 1 (≥50%), 2 (≥60%), 3 (≥80% confiance)

---

## 📜 Scripts Disponibles

### `model_improved.py`
**Entraînement du modèle amélioré**
- Extrait les caractéristiques
- Normalise les données
- Augmente les données
- Entraîne le modèle
- Sauvegarde le modèle et le scaler

### `predict_improved.py`
**Prédiction sur un fichier audio**
- Charge le modèle et le scaler
- Extrait les caractéristiques
- Fait la prédiction
- Affiche les résultats

### `predict_full_audio.py`
**Analyse d'un fichier audio complet**
- Découpe le fichier en segments
- Prédit chaque segment
- Fusionne les segments consécutifs
- Génère un CSV avec les résultats

### `main.py`
**Script principal avec menu interactif**
- Interface simple pour lancer toutes les étapes
- Usage : `python main.py` ou `python main.py [numéro]`

### `slicing.py`
**Découpage d'un fichier audio**
- Découpe un fichier audio en segments basés sur des annotations
- Génère les fichiers dans `data/slices/`
- Crée `data/annotation.csv` automatiquement

### `annotation.py`
**Création du fichier d'annotations**
- Convertit des données brutes en format CSV

---

## 🔍 Dépannage

### Erreur : "No librosa.feature attribute chroma"
**Solution :** Le script détecte automatiquement les fonctions disponibles. Si chroma n'est pas disponible, des zéros seront utilisés à la place.

### Erreur : "The least populated class has only 1 member"
**Solution :** Chaque classe doit avoir au moins 2 échantillons. Ajoutez plus de données pour les classes minoritaires.

### Erreur : "File not found"
**Solution :** Vérifiez que :
- Les fichiers audio sont dans le dossier `data/slices/`
- Les noms dans `data/annotation.csv` correspondent exactement aux noms des fichiers
- Les chemins sont corrects

### Performance Faible
**Solutions possibles :**
1. **Ajouter plus de données** : Le modèle a besoin d'au moins 50-100 échantillons par classe pour de bonnes performances
2. **Équilibrer les classes** : Les classes déséquilibrées peuvent causer des problèmes
3. **Vérifier la qualité des annotations** : Des annotations incorrectes dégradent les performances
4. **Ajuster les hyperparamètres** : Modifier le learning rate, batch size, etc.

### Modèle qui Surapprend
**Le modèle inclut déjà :**
- Dropout (0.2-0.5)
- BatchNormalization
- Early stopping
- Augmentation de données

**Si le surapprentissage persiste :**
- Ajoutez plus de données
- Augmentez le dropout
- Réduisez la taille du modèle

---

## 📈 Améliorations Futures

- [ ] Support pour les architectures CNN 2D (spectrogrammes complets)
- [ ] Support pour LSTM/BiLSTM (dépendances temporelles)
- [ ] Transfer learning avec modèles pré-entraînés
- [ ] Interface web pour l'annotation
- [ ] Validation croisée k-fold
- [ ] Recherche d'hyperparamètres automatisée
- [ ] Support pour l'apprentissage continu (fine-tuning)

---

## 📝 Notes Importantes

1. **Données limitées** : Avec seulement 27-29 échantillons, les performances seront limitées. Collectez plus de données pour de meilleurs résultats.

2. **Classes déséquilibrées** : Si certaines classes ont beaucoup plus d'échantillons que d'autres, le modèle peut être biaisé. Essayez d'équilibrer les classes.

3. **Qualité des annotations** : La qualité du modèle dépend directement de la qualité des annotations. Vérifiez que les labels sont corrects.

4. **Normalisation** : Le scaler (`models/scaler.pkl`) doit être utilisé avec le même modèle. Si vous réentraînez le modèle, régénérez le scaler.

5. **Compatibilité** : Le modèle sauvegardé (`model_improved.h5`) est compatible avec TensorFlow 2.x.

---

## 📧 Support

Pour toute question ou problème, consultez :
- Le fichier `AMELIORATIONS.md` pour les détails techniques des améliorations
- Les commentaires dans les scripts Python
- La documentation TensorFlow : https://www.tensorflow.org/
- La documentation librosa : https://librosa.org/

---

**Dernière mise à jour :** Décembre 2025
