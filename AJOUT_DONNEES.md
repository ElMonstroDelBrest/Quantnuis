# Guide : Ajouter de Nouvelles Données

Ce guide explique comment ajouter de nouvelles données à votre base d'entraînement.

## 📋 Prérequis

- Un fichier CSV avec les annotations brutes (format: Start, End, Label, Reliability)
- Le fichier audio source correspondant

## 🚀 Workflow Complet

### Étape 1 : Préparer le fichier d'annotations brutes

Vous avez deux options :

#### Option A : Utiliser `annotation.py` (si les données sont dans le script)

1. Ouvrez `annotation.py`
2. Modifiez la variable `raw_data` avec vos nouvelles annotations
3. Lancez :
   ```bash
   python main.py 1
   ```
   Cela génère `data/raw/annotations_raw.csv`

#### Option B : Créer manuellement le CSV

Créez un fichier CSV dans `data/raw/` avec le format suivant :

```csv
Start,End,Label,Reliability
00:09:34,00:10:12,1,3
00:11:30,00:11:43,2,3
00:12:09,00:12:24,3,1
```

**Format des colonnes :**
- `Start` : Timestamp de début (format HH:MM:SS)
- `End` : Timestamp de fin (format HH:MM:SS)
- `Label` : Classe/label (entier, ex: 1, 2, 3, 4)
- `Reliability` : Fiabilité (1-3, où 3 = très fiable)

**Important :** Le fichier doit être nommé `annotations_raw.csv` ou `*_annotations_raw.csv` et placé dans `data/raw/`

### Étape 2 : Découper le fichier audio

1. Placez votre fichier audio dans le répertoire du projet (ou notez son chemin)
2. Lancez :
   ```bash
   python main.py 2
   ```
   Ou directement :
   ```bash
   python slicing.py [chemin_vers_fichier_audio]
   ```

**Ce que fait le script :**
- ✅ Lit automatiquement le fichier `annotations_raw.csv` le plus récent dans `data/raw/`
- ✅ Découpe le fichier audio en segments selon les timestamps
- ✅ **Ajoute** les nouveaux slices dans `data/slices/` (sans écraser les existants)
- ✅ **Ajoute** les nouvelles annotations dans `data/annotation.csv` (sans doublons)
- ✅ Numérote automatiquement les slices en continuant la séquence existante
- ✅ Ignore les doublons (même nom de fichier)

### Étape 3 : Vérifier les données

Vérifiez que tout s'est bien passé :

```bash
# Voir le nombre de slices
ls data/slices/*.wav | wc -l

# Voir les annotations
head data/annotation.csv
```

### Étape 4 : Réentraîner le modèle (optionnel mais recommandé)

Une fois les nouvelles données ajoutées, réentraînez le modèle :

```bash
python main.py 3
```

Cela va :
- Extraire les caractéristiques de **tous** les fichiers dans `data/slices/`
- Utiliser **toutes** les annotations de `data/annotation.csv`
- Entraîner un nouveau modèle avec les données augmentées

## 📝 Exemple Complet

Supposons que vous avez :
- Un fichier audio : `nouvel_audio.wav`
- Un CSV avec des annotations : `mes_annotations.csv`

**Workflow :**

```bash
# 1. Copier le CSV dans data/raw/
cp mes_annotations.csv data/raw/annotations_raw.csv

# 2. Découper l'audio
python main.py 2
# Entrez le chemin : nouvel_audio.wav

# 3. Vérifier
ls data/slices/ | tail -5  # Voir les derniers slices ajoutés

# 4. Réentraîner
python main.py 3
```

## ⚠️ Points Importants

1. **Pas de doublons** : Le système détecte automatiquement les doublons par nom de fichier et les ignore
2. **Numérotation continue** : Les nouveaux slices continuent la numérotation existante (slice_028.wav, slice_029.wav, etc.)
3. **Fusion automatique** : Toutes les annotations sont fusionnées dans `data/annotation.csv`
4. **Pas d'écrasement** : Les données existantes ne sont jamais écrasées, seulement ajoutées

## 🔄 Si vous avez plusieurs fichiers CSV

Si vous avez plusieurs fichiers CSV à traiter :

1. Renommez-les avec un suffixe : `annotations_raw_1.csv`, `annotations_raw_2.csv`, etc.
2. Placez-les tous dans `data/raw/`
3. Pour chaque fichier :
   - Renommez-le temporairement en `annotations_raw.csv`
   - Lancez `python main.py 2` avec le fichier audio correspondant
   - Le script utilisera automatiquement le fichier le plus récent

## 📊 Vérifier l'état de votre base de données

Pour voir combien de slices vous avez par classe :

```python
import pandas as pd
df = pd.read_csv('data/annotation.csv')
print(df['label'].value_counts().sort_index())
```

## 🆘 Dépannage

**Problème : Les slices ne sont pas ajoutés**
- Vérifiez que le fichier `annotations_raw.csv` est dans `data/raw/`
- Vérifiez le format du CSV (colonnes : Start, End, Label, Reliability)
- Vérifiez que les timestamps sont au format HH:MM:SS

**Problème : Des doublons sont créés**
- Le système devrait les détecter automatiquement
- Vérifiez les noms de fichiers dans `data/slices/`
- Si nécessaire, supprimez manuellement les doublons

**Problème : Le modèle ne s'améliore pas**
- Vérifiez que vous avez assez de données (minimum 50-100 échantillons par classe recommandé)
- Vérifiez la qualité des annotations
- Vérifiez que les classes sont équilibrées
