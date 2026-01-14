# 🎵 Gestion de Base de Données Audio

Outils pour gérer une base de données de segments audio annotés.

## 📁 Structure

```
Quantnuis/
├── data/
│   ├── slices/           ← Fichiers audio .wav
│   ├── annotation.csv    ← Annotations (labels)
│   └── features.csv      ← Caractéristiques extraites
├── slice_manager.py      ← Gestion de la base
├── data_augmentation.py  ← Augmentation des données
├── feature_extraction.py ← Extraction de features
└── slicing.py            ← Découpage de fichiers audio
```

## 🔧 Installation

```bash
pip install librosa soundfile pandas numpy pydub
```

## 📋 Format des Annotations

Le fichier `data/annotation.csv` contient :

```csv
nfile,length,label,reliability
slice_001.wav,38,1,3
slice_002.wav,13,2,3
```

| Colonne | Description |
|---------|-------------|
| `nfile` | Nom du fichier audio |
| `length` | Durée en secondes |
| `label` | Classe (1, 2, ...) |
| `reliability` | Fiabilité (1-3) |

## 🛠️ Scripts

### `slice_manager.py` - Gestionnaire de base

```bash
python slice_manager.py          # Afficher le statut
python slice_manager.py add DIR  # Ajouter des slices depuis un dossier
```

### `data_augmentation.py` - Augmentation

Crée des versions modifiées des slices pour augmenter le dataset.

```bash
python data_augmentation.py              # Menu interactif
python data_augmentation.py status       # Voir les stats
python data_augmentation.py augment      # Augmenter tout
python data_augmentation.py augment 1    # Augmenter label 1 uniquement
```

**Augmentations disponibles :**
- Bruit gaussien
- Changement de vitesse (lent/rapide)
- Changement de pitch (haut/bas)

### `feature_extraction.py` - Extraction de caractéristiques

Extrait ~115 caractéristiques audio par fichier pour le machine learning.

```bash
python feature_extraction.py              # Extraire tout
python feature_extraction.py status       # Voir le statut
python feature_extraction.py --label 1    # Extraire seulement label 1
python feature_extraction.py --force      # Réextraire tout
```

**Caractéristiques extraites :**
| Catégorie | Nombre | Description |
|-----------|--------|-------------|
| Base | 4 | RMS (volume), ZCR |
| Spectral | 10 | Centroid, bandwidth, rolloff, flatness, contrast |
| Harmonic/Percussive | 5 | Séparation sons percussifs/harmoniques |
| MFCCs | 80 | 40 coefficients (mean + std) - timbre |
| Chroma | 14 | Tonalité (12 notes + global) |
| Autres | 4 | Tempo, durée, énergie, amplitude max |

### `slicing.py` - Découpage

Découpe un fichier audio long selon des annotations.

```bash
python slicing.py audio.wav annotations.csv
```

**Format du CSV d'entrée :**
```csv
Start,End,Label,Reliability
00:09:34,00:10:12,1,3
00:11:30,00:11:43,2,3
```

## 📊 Exemple d'utilisation

1. **Voir le statut actuel :**
   ```bash
   python slice_manager.py
   ```

2. **Découper un nouveau fichier audio :**
   ```bash
   python slicing.py enregistrement.wav mes_annotations.csv
   ```

3. **Augmenter les données (label minoritaire) :**
   ```bash
   python data_augmentation.py augment 1
   ```

4. **Extraire les features pour ML :**
   ```bash
   python feature_extraction.py
   ```
   → Génère `data/features.csv` avec ~115 caractéristiques par fichier

---

**Structure de données :** Tous les fichiers audio sont dans `data/slices/`, toutes les annotations dans `data/annotation.csv`.
