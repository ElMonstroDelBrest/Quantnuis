#!/usr/bin/env python3
"""
================================================================================
                    PRÉDICTION SUR UN FICHIER AUDIO
================================================================================

Ce script utilise le modèle entraîné pour prédire si un fichier audio est
"Bruyant" ou "Normal".

┌─────────────────────────────────────────────────────────────────────────────┐
│  FONCTIONNEMENT GÉNÉRAL                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. L'utilisateur fournit un fichier audio (.wav, .mp3, etc.)               │
│                                                                             │
│  2. Le script charge le modèle TensorFlow entraîné précédemment             │
│                                                                             │
│  3. Les features audio sont extraites (MFCC, spectral, etc.)                │
│     ⚠️ IMPORTANT: Ce sont les MÊMES features que lors de l'entraînement     │
│                                                                             │
│  4. Les features sont standardisées avec le MÊME scaler                     │
│                                                                             │
│  5. Le modèle prédit une probabilité entre 0 et 1                           │
│     - Proche de 0 → Audio BRUYANT                                           │
│     - Proche de 1 → Audio NORMAL                                            │
│                                                                             │
│  6. Le résultat est affiché avec un pourcentage de confiance                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    python predict.py chemin/vers/audio.wav
    python predict.py chemin/vers/audio.mp3
    python predict.py --test  # Test sur un fichier du dataset

Exemple:
    python predict.py data/slices/slice_001.wav

================================================================================
"""

# ==============================================================================
# ██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗███████╗
# ██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝
# ██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ███████╗
# ██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ╚════██║
# ██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ███████║
# ╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
# ==============================================================================

# ------------------------------------------------------------------------------
# os : Module standard Python pour interagir avec le système d'exploitation
# - os.path.exists() : Vérifie si un fichier/dossier existe
# - Utilisé pour valider les chemins avant de charger les fichiers
# ------------------------------------------------------------------------------
import os

# ------------------------------------------------------------------------------
# sys : Module standard Python pour les paramètres système
# - sys.argv : Liste des arguments passés en ligne de commande
#   sys.argv[0] = nom du script ("predict.py")
#   sys.argv[1] = premier argument (chemin du fichier audio)
# - sys.exit() : Quitte le programme avec un code de sortie
# ------------------------------------------------------------------------------
import sys

# ------------------------------------------------------------------------------
# numpy : Bibliothèque de calcul numérique
# - np.array() : Crée des tableaux multidimensionnels
# - np.mean(), np.std() : Calculs statistiques sur les features
# - Essentiel pour manipuler les données audio et les features
# ------------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------------------
# warnings : Gestion des avertissements Python
# - filterwarnings('ignore') : Supprime les warnings pour une sortie propre
# - Librosa génère beaucoup de warnings (dépréciation, etc.)
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# librosa : Bibliothèque d'analyse audio (même que feature_extraction.py)
# - librosa.load() : Charge un fichier audio en tableau numpy
# - librosa.feature.* : Extraction des caractéristiques audio
#   ├── mfcc : Coefficients cepstraux (empreinte du son)
#   ├── spectral_centroid : Centre de gravité du spectre
#   ├── spectral_bandwidth : Largeur du spectre
#   ├── chroma : Notes musicales
#   └── etc.
# 
# CRITIQUE: Les features extraites DOIVENT être identiques à celles utilisées
#           lors de l'entraînement, sinon le modèle ne fonctionnera pas !
# ------------------------------------------------------------------------------
import librosa

# ------------------------------------------------------------------------------
# joblib : Sérialisation d'objets Python
# - joblib.load() : Charge le StandardScaler sauvegardé
# - Le scaler contient les moyennes et écarts-types calculés sur le dataset
#   d'entraînement, nécessaires pour standardiser les nouvelles données
# ------------------------------------------------------------------------------
import joblib

# ------------------------------------------------------------------------------
# tensorflow : Framework de deep learning (Google)
# - tf.keras.models.load_model() : Charge le modèle .keras sauvegardé
# - model.predict() : Fait l'inférence sur les nouvelles données
# 
# NOTE: TensorFlow peut être lent à charger (~2-5 secondes au premier import)
#       car il initialise CUDA/cuDNN si disponible
# ------------------------------------------------------------------------------
import tensorflow as tf


# ==============================================================================
#  ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗ 
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝ 
# ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
# ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
# ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝ 
# ==============================================================================
# Ces chemins pointent vers les fichiers générés par train_model.py
# Si ces fichiers n'existent pas, le script affichera une erreur explicative
# ==============================================================================

# ------------------------------------------------------------------------------
# MODEL_PATH : Chemin vers le modèle Keras sauvegardé
# 
# Le fichier .keras contient :
# - L'architecture du réseau (couches, neurones, activations)
# - Les poids entraînés (les valeurs numériques apprises)
# - La configuration de compilation (optimizer, loss, metrics)
# 
# Format .keras = format natif TensorFlow (recommandé depuis TF 2.12)
# ------------------------------------------------------------------------------
MODEL_PATH = "models/model_final.keras"

# ------------------------------------------------------------------------------
# SCALER_PATH : Chemin vers le StandardScaler sauvegardé
# 
# Le scaler contient les statistiques du dataset d'entraînement :
# - mean_ : Moyenne de chaque feature
# - scale_ : Écart-type de chaque feature
# 
# POURQUOI C'EST CRITIQUE ?
# ┌──────────────────────────────────────────────────────────────────┐
# │ Lors de l'entraînement, on calcule :                             │
# │   X_scaled = (X - mean_train) / std_train                        │
# │                                                                  │
# │ Pour la prédiction, on DOIT utiliser les MÊMES mean et std :     │
# │   X_new_scaled = (X_new - mean_train) / std_train                │
# │                                                                  │
# │ Si on recalcule mean/std sur X_new, les valeurs seront           │
# │ différentes et le modèle ne comprendra pas les données !         │
# └──────────────────────────────────────────────────────────────────┘
# ------------------------------------------------------------------------------
SCALER_PATH = "models/scaler_final.pkl"

# ------------------------------------------------------------------------------
# FEATURES_FILE : Liste des features utilisées par le modèle
# 
# Ce fichier texte contient une feature par ligne, dans l'ORDRE exact
# attendu par le modèle. Exemple :
#   mfcc_39_mean
#   mfcc_11_std
#   mfcc_37_mean
#   ...
# 
# L'ORDRE EST CRUCIAL : si on inverse deux features, le modèle recevra
# des valeurs dans le mauvais neurone et les prédictions seront fausses.
# ------------------------------------------------------------------------------
FEATURES_FILE = "models/features_used.txt"

# ------------------------------------------------------------------------------
# SAMPLE_RATE : Fréquence d'échantillonnage audio
# 
# 22050 Hz = Standard pour l'analyse audio (qualité suffisante pour la voix
# et la musique, tout en étant 2x plus léger que le CD quality 44100 Hz)
# 
# IMPORTANT : Doit être identique à feature_extraction.py
# Si le fichier source est en 44100 Hz, librosa le rééchantillonnera
# automatiquement à 22050 Hz.
# ------------------------------------------------------------------------------
SAMPLE_RATE = 22050


# ==============================================================================
#  ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗██╗   ██╗██████╗ ███████╗
# ██╔════╝██╔═══██╗██║   ██║██║     ██╔════╝██║   ██║██╔══██╗██╔════╝
# ██║     ██║   ██║██║   ██║██║     █████╗  ██║   ██║██████╔╝███████╗
# ██║     ██║   ██║██║   ██║██║     ██╔══╝  ██║   ██║██╔══██╗╚════██║
# ╚██████╗╚██████╔╝╚██████╔╝███████╗███████╗╚██████╔╝██║  ██║███████║
#  ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
#                    AFFICHAGE COLORÉ DANS LE TERMINAL
# ==============================================================================
# Les codes ANSI permettent de colorer le texte dans le terminal
# Format : \033[XXm où XX est le code couleur
# Ces codes sont supportés par la plupart des terminaux modernes
# ==============================================================================

class Colors:
    """
    Classe contenant les codes ANSI pour colorer le texte dans le terminal.
    
    COMMENT ÇA MARCHE ?
    ───────────────────
    Les séquences d'échappement ANSI sont des caractères spéciaux interprétés
    par le terminal pour modifier l'affichage du texte.
    
    Structure : \033[XXm
    - \033 (ou \x1b) : Caractère d'échappement (ESC)
    - [ : Début de la séquence
    - XX : Code de style/couleur
    - m : Fin de la séquence
    
    USAGE :
    ───────
    print(f"{Colors.GREEN}Texte vert{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}Erreur en gras rouge{Colors.END}")
    
    IMPORTANT : Toujours terminer par Colors.END pour réinitialiser le style,
                sinon tout le texte suivant gardera le style.
    """
    
    # --- COULEURS DE TEXTE ---
    # Ces codes changent la couleur du texte affiché
    
    CYAN = '\033[96m'      # Cyan clair   - Pour les informations, titres
    GREEN = '\033[92m'     # Vert clair   - Pour les succès, confirmations
    YELLOW = '\033[93m'    # Jaune clair  - Pour les avertissements
    RED = '\033[91m'       # Rouge clair  - Pour les erreurs
    
    # --- STYLES DE TEXTE ---
    # Ces codes modifient le style sans changer la couleur
    
    BOLD = '\033[1m'       # Gras         - Pour mettre en évidence
    DIM = '\033[2m'        # Atténué      - Pour les infos secondaires
    
    # --- RÉINITIALISATION ---
    # Ce code remet TOUS les styles à leur valeur par défaut
    
    END = '\033[0m'        # Reset        - OBLIGATOIRE après chaque style


# ------------------------------------------------------------------------------
# FONCTIONS D'AFFICHAGE
# Ces fonctions encapsulent la logique d'affichage pour garder le code propre
# ------------------------------------------------------------------------------

def print_header(title: str):
    """
    Affiche un en-tête de section formaté.
    
    Résultat visuel :
    ──────────────────────────────────────────────────
      TITRE DE LA SECTION
    ──────────────────────────────────────────────────
    
    Paramètres:
        title (str): Le titre à afficher (sera converti en MAJUSCULES)
    
    Notes:
        - La ligne de séparation fait 50 caractères de large
        - Le titre est automatiquement mis en majuscules pour l'impact visuel
        - Une ligne vide est ajoutée avant pour la lisibilité
    """
    width = 50  # Largeur fixe de l'en-tête en caractères
    print()     # Ligne vide pour séparer des sections précédentes
    
    # Ligne de séparation supérieure (caractère Unicode ─)
    print(f"{Colors.CYAN}{Colors.BOLD}{'─' * width}{Colors.END}")
    
    # Titre en majuscules avec indentation de 2 espaces
    print(f"{Colors.CYAN}{Colors.BOLD}  {title.upper()}{Colors.END}")
    
    # Ligne de séparation inférieure
    print(f"{Colors.CYAN}{'─' * width}{Colors.END}")


def print_success(msg: str):
    """
    Affiche un message de succès avec le préfixe [OK] en vert.
    
    Résultat visuel :
    [OK] Message de succès
    
    Paramètres:
        msg (str): Le message à afficher
    
    Usage:
        print_success("Modèle chargé avec succès")
        # Affiche : [OK] Modèle chargé avec succès
    """
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")


def print_info(msg: str):
    """
    Affiche un message d'information avec le préfixe [i] en cyan.
    
    Résultat visuel :
    [i] Message informatif
    
    Paramètres:
        msg (str): Le message à afficher
    
    Usage:
        print_info("Chargement en cours...")
        # Affiche : [i] Chargement en cours...
    """
    print(f"{Colors.CYAN}[i]{Colors.END} {msg}")


def print_warning(msg: str):
    """
    Affiche un avertissement avec le préfixe [!] en jaune.
    
    Résultat visuel :
    [!] Message d'avertissement
    
    Paramètres:
        msg (str): Le message à afficher
    
    Usage:
        print_warning("Feature manquante, valeur par défaut utilisée")
        # Affiche : [!] Feature manquante, valeur par défaut utilisée
    """
    print(f"{Colors.YELLOW}[!]{Colors.END} {msg}")


def print_error(msg: str):
    """
    Affiche un message d'erreur avec le préfixe [ERREUR] en rouge.
    
    Résultat visuel :
    [ERREUR] Description de l'erreur
    
    Paramètres:
        msg (str): Le message d'erreur à afficher
    
    Usage:
        print_error("Fichier non trouvé: audio.wav")
        # Affiche : [ERREUR] Fichier non trouvé: audio.wav
    """
    print(f"{Colors.RED}[ERREUR]{Colors.END} {msg}")


# ==============================================================================
# ███████╗███████╗ █████╗ ████████╗██╗   ██╗██████╗ ███████╗███████╗
# ██╔════╝██╔════╝██╔══██╗╚══██╔══╝██║   ██║██╔══██╗██╔════╝██╔════╝
# █████╗  █████╗  ███████║   ██║   ██║   ██║██████╔╝█████╗  ███████╗
# ██╔══╝  ██╔══╝  ██╔══██║   ██║   ██║   ██║██╔══██╗██╔══╝  ╚════██║
# ██║     ███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║███████╗███████║
# ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
#                    EXTRACTION DES FEATURES AUDIO
# ==============================================================================

def extract_features(file_path: str, feature_names: list) -> dict:
    """
    Extrait les caractéristiques audio d'un fichier pour la prédiction.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PRINCIPE DE L'EXTRACTION DE FEATURES                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Un fichier audio est une séquence de valeurs (échantillons) qui        │
    │  représentent l'amplitude du son au fil du temps.                       │
    │                                                                         │
    │  Exemple : [0.1, 0.3, 0.5, 0.3, 0.1, -0.2, -0.4, ...]                   │
    │            (22050 valeurs par seconde si sr=22050)                      │
    │                                                                         │
    │  Le problème : un fichier de 3 secondes = 66150 valeurs !               │
    │  Le modèle ne peut pas traiter autant de données directement.           │
    │                                                                         │
    │  La solution : extraire des FEATURES (caractéristiques résumées)        │
    │  qui capturent l'essence du son en quelques dizaines de valeurs.        │
    │                                                                         │
    │  Analogie : au lieu de décrire quelqu'un pixel par pixel,               │
    │  on dit "homme, 30 ans, cheveux bruns, 1m80".                           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TYPES DE FEATURES EXTRAITES                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. FEATURES TEMPORELLES (analysent le signal brut)                     │
    │     ├── RMS : Volume moyen du son                                       │
    │     └── ZCR : Combien de fois le signal passe par 0 (bruit = beaucoup) │
    │                                                                         │
    │  2. FEATURES SPECTRALES (analysent les fréquences)                      │
    │     ├── Spectral Centroid : Fréquence "moyenne" (brillant vs sourd)    │
    │     ├── Spectral Bandwidth : Étendue des fréquences                     │
    │     ├── Spectral Rolloff : Fréquence max significative                  │
    │     ├── Spectral Flatness : Bruit blanc vs son tonal                    │
    │     └── Spectral Contrast : Différence pics/creux du spectre            │
    │                                                                         │
    │  3. MFCC (Mel-Frequency Cepstral Coefficients)                          │
    │     └── 40 coefficients qui forment une "empreinte" du son              │
    │         Très utilisés en reconnaissance vocale et audio                 │
    │                                                                         │
    │  4. CHROMA (analysent les notes musicales)                              │
    │     └── 12 valeurs (une par note : C, C#, D, D#, E, F, F#, G, G#, A,   │
    │         A#, B) qui représentent l'énergie dans chaque note              │
    │                                                                         │
    │  5. FEATURES HARMONIQUES/PERCUSSIVES                                    │
    │     ├── Harmonic : Partie "musicale" du son (voix, instruments)         │
    │     └── Percussive : Partie "bruits" (claquements, impacts)             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Paramètres:
        file_path (str): Chemin vers le fichier audio à analyser
                         Formats supportés : .wav, .mp3, .flac, .ogg, .m4a
        
        feature_names (list): Liste des noms de features à retourner
                              Ces noms doivent correspondre aux features
                              utilisées lors de l'entraînement du modèle
    
    Retourne:
        dict: Dictionnaire {nom_feature: valeur}
              Exemple : {'mfcc_1_mean': 0.234, 'spectral_centroid_mean': 1523.4, ...}
    
    Raises:
        Exception: Si le fichier ne peut pas être lu (format invalide, corrompu, etc.)
    
    ⚠️ ATTENTION CRITIQUE :
    Cette fonction DOIT extraire les features de la MÊME MANIÈRE que
    feature_extraction.py, sinon les prédictions seront incorrectes !
    """
    
    # ==========================================================================
    # ÉTAPE 1 : CHARGEMENT DU FICHIER AUDIO
    # ==========================================================================
    # librosa.load() fait plusieurs choses :
    # 1. Ouvre le fichier (supporte wav, mp3, flac, ogg, etc. via ffmpeg)
    # 2. Convertit en mono si stéréo (moyenne des canaux)
    # 3. Rééchantillonne à la fréquence demandée (sr=SAMPLE_RATE)
    # 4. Retourne un tableau numpy de floats entre -1 et 1
    # ==========================================================================
    
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # y  = tableau numpy des échantillons audio (forme : (n_samples,))
    # sr = sample rate effectif (devrait être = SAMPLE_RATE)
    
    # --------------------------------------------------------------------------
    # NORMALISATION DU SIGNAL
    # Ramène les valeurs entre -1 et 1 pour avoir une amplitude cohérente
    # Cela permet de comparer des enregistrements faits à différents volumes
    # --------------------------------------------------------------------------
    y = librosa.util.normalize(y)
    
    # Dictionnaire qui contiendra toutes les features extraites
    features = {}
    
    # ==========================================================================
    # ÉTAPE 2 : FEATURES TEMPORELLES
    # Ces features analysent directement le signal audio dans le temps
    # ==========================================================================
    
    # --------------------------------------------------------------------------
    # RMS (Root Mean Square) - VOLUME MOYEN
    # 
    # Formule : RMS = sqrt(mean(signal²))
    # 
    # Interprétation :
    # - RMS élevé → Son fort
    # - RMS faible → Son faible
    # 
    # Le RMS est calculé par fenêtres (frames) pour suivre l'évolution
    # du volume au cours du temps. On prend ensuite la moyenne et l'écart-type.
    # --------------------------------------------------------------------------
    rms = librosa.feature.rms(y=y)[0]  # [0] car retourne un array 2D (1, n_frames)
    features['rms_mean'] = float(np.mean(rms))  # Volume moyen global
    features['rms_std'] = float(np.std(rms))    # Variation du volume (dynamique)
    
    # --------------------------------------------------------------------------
    # ZCR (Zero Crossing Rate) - TAUX DE PASSAGE PAR ZÉRO
    # 
    # Compte combien de fois le signal passe de positif à négatif (ou inverse)
    # par unité de temps.
    # 
    # Interprétation :
    # - ZCR élevé → Son bruité, percussif (beaucoup d'oscillations rapides)
    # - ZCR faible → Son tonal, musical (oscillations régulières)
    # 
    # Très utile pour distinguer :
    # - Voix parlée (ZCR moyen) vs musique (ZCR bas) vs bruit (ZCR haut)
    # --------------------------------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # ==========================================================================
    # ÉTAPE 3 : FEATURES SPECTRALES
    # Ces features analysent la distribution des fréquences dans le signal
    # (via la Transformée de Fourier)
    # ==========================================================================
    
    # --------------------------------------------------------------------------
    # SPECTRAL CENTROID - CENTRE DE GRAVITÉ DU SPECTRE
    # 
    # C'est la fréquence "moyenne" pondérée par l'énergie.
    # 
    # Analogie visuelle :
    # Si on représente le spectre comme un tas de sable sur une balance,
    # le centroid est le point d'équilibre.
    # 
    # Interprétation :
    # - Centroid élevé → Son brillant, aigu (violon, cymbale)
    # - Centroid bas → Son sourd, grave (basse, grosse caisse)
    # --------------------------------------------------------------------------
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    
    # --------------------------------------------------------------------------
    # SPECTRAL BANDWIDTH - LARGEUR DU SPECTRE
    # 
    # Mesure l'étendue des fréquences autour du centroid.
    # 
    # Interprétation :
    # - Bandwidth élevé → Son riche en harmoniques (orchestre, bruit)
    # - Bandwidth bas → Son pur, simple (diapason, sinus)
    # --------------------------------------------------------------------------
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
    features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
    
    # --------------------------------------------------------------------------
    # SPECTRAL ROLLOFF - FRÉQUENCE DE COUPURE
    # 
    # Fréquence en dessous de laquelle se trouve 85% de l'énergie spectrale.
    # Permet de savoir si le son est plutôt grave ou aigu.
    # 
    # Interprétation :
    # - Rolloff élevé → Énergie dans les hautes fréquences
    # - Rolloff bas → Énergie concentrée dans les basses fréquences
    # --------------------------------------------------------------------------
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
    
    # --------------------------------------------------------------------------
    # SPECTRAL FLATNESS - PLANÉITÉ DU SPECTRE
    # 
    # Ratio entre moyenne géométrique et moyenne arithmétique du spectre.
    # Mesure à quel point le spectre est "plat" (bruit blanc) vs "peaky" (tonal).
    # 
    # Valeurs :
    # - Proche de 1 → Bruit blanc (toutes fréquences égales)
    # - Proche de 0 → Son tonal (pics à certaines fréquences)
    # --------------------------------------------------------------------------
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
    features['spectral_flatness_std'] = float(np.std(spectral_flatness))
    
    # --------------------------------------------------------------------------
    # SPECTRAL CONTRAST - CONTRASTE SPECTRAL
    # 
    # Différence entre les pics et les vallées dans chaque bande de fréquence.
    # Utile pour distinguer différents types de sons.
    # 
    # Interprétation :
    # - Contraste élevé → Sons avec des harmoniques claires (instruments)
    # - Contraste bas → Sons bruités, sans structure claire
    # --------------------------------------------------------------------------
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
    features['spectral_contrast_std'] = float(np.std(spectral_contrast))
    
    # ==========================================================================
    # ÉTAPE 4 : SÉPARATION HARMONIQUE / PERCUSSIVE
    # Librosa peut séparer le son en deux composantes :
    # - Harmonique : parties musicales, tonales (voix, instruments mélodiques)
    # - Percussive : parties rythmiques, bruits (drums, claquements)
    # ==========================================================================
    
    harmonic, percussive = librosa.effects.hpss(y)
    # harmonic = signal filtré ne gardant que les composantes harmoniques
    # percussive = signal filtré ne gardant que les composantes percussives
    
    features['harm_mean'] = float(np.mean(np.abs(harmonic)))
    features['harm_std'] = float(np.std(harmonic))
    features['perc_mean'] = float(np.mean(np.abs(percussive)))
    features['perc_std'] = float(np.std(percussive))
    
    # Ratio harmonique/percussif : indique si le son est plutôt musical ou bruité
    # Le + 1e-10 évite la division par zéro si percussive est nul
    features['harm_perc_ratio'] = float(
        np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-10)
    )
    
    # ==========================================================================
    # ÉTAPE 5 : MFCC (Mel-Frequency Cepstral Coefficients)
    # ==========================================================================
    # 
    # LES MFCC SONT LES FEATURES LES PLUS IMPORTANTES POUR L'AUDIO !
    # 
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  QU'EST-CE QUE LES MFCC ?                                          │
    # ├─────────────────────────────────────────────────────────────────────┤
    # │                                                                     │
    # │  Les MFCC sont une représentation compacte du spectre audio qui    │
    # │  imite la perception humaine des fréquences.                       │
    # │                                                                     │
    # │  Étapes de calcul :                                                │
    # │  1. Découper le signal en fenêtres (ex: 23ms)                      │
    # │  2. Calculer le spectre de chaque fenêtre (FFT)                    │
    # │  3. Appliquer un banc de filtres Mel (échelle perceptive)          │
    # │  4. Prendre le log de l'énergie dans chaque filtre                 │
    # │  5. Appliquer une DCT (Discrete Cosine Transform)                  │
    # │                                                                     │
    # │  Résultat : 40 coefficients qui forment une "empreinte" du son     │
    # │                                                                     │
    # │  Les premiers MFCC capturent :                                     │
    # │  - MFCC 1 : Énergie globale (volume)                               │
    # │  - MFCC 2-13 : Forme générale du spectre (timbre)                  │
    # │  - MFCC 14+ : Détails fins (texture)                               │
    # │                                                                     │
    # └─────────────────────────────────────────────────────────────────────┘
    # ==========================================================================
    
    # Extraction de 40 coefficients MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # mfccs.shape = (40, n_frames) : 40 coefficients × nombre de fenêtres
    
    # Pour chaque coefficient, on calcule la moyenne et l'écart-type
    # sur toutes les fenêtres temporelles
    for i in range(40):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
    # Cela donne 80 features MFCC (40 moyennes + 40 écarts-types)
    
    # ==========================================================================
    # ÉTAPE 6 : CHROMA (NOTES MUSICALES)
    # ==========================================================================
    # 
    # Les chroma features représentent l'énergie dans les 12 notes de la gamme
    # chromatique : C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    # 
    # Utile pour :
    # - Détection de tonalité
    # - Reconnaissance de morceaux
    # - Analyse harmonique
    # ==========================================================================
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # chroma.shape = (12, n_frames) : 12 notes × nombre de fenêtres
    
    features['chroma_mean'] = float(np.mean(chroma))  # Moyenne globale
    features['chroma_std'] = float(np.std(chroma))    # Variation globale
    
    # Moyenne par note (12 valeurs)
    for i in range(12):
        features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
    
    # ==========================================================================
    # ÉTAPE 7 : TEMPO
    # ==========================================================================
    # 
    # Le tempo est le nombre de battements par minute (BPM).
    # Librosa l'estime en détectant les pics d'énergie rythmique.
    # 
    # Note : l'estimation peut échouer pour des sons non musicaux
    # ==========================================================================
    
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
    except:
        # Si l'estimation échoue, on met 0 par défaut
        features['tempo'] = 0.0
    
    # ==========================================================================
    # ÉTAPE 8 : FEATURES GLOBALES
    # ==========================================================================
    
    # Énergie totale : somme des carrés de tous les échantillons
    features['energy'] = float(np.sum(y**2))
    
    # Amplitude maximale : pic le plus élevé du signal
    features['max_amplitude'] = float(np.max(np.abs(y)))
    
    # ==========================================================================
    # ÉTAPE 9 : SÉLECTION DES FEATURES DEMANDÉES
    # ==========================================================================
    # 
    # Le modèle n'utilise pas toutes les features, seulement celles
    # identifiées comme importantes lors de l'analyse (feature_importance).
    # 
    # On ne garde que les features listées dans feature_names.
    # ==========================================================================
    
    selected_features = {}
    
    for name in feature_names:
        if name in features:
            # La feature existe, on la garde
            selected_features[name] = features[name]
        else:
            # La feature n'a pas été extraite (erreur de nom ?)
            # On met 0 par défaut et on avertit l'utilisateur
            print_warning(f"Feature '{name}' non trouvée, mise à 0")
            selected_features[name] = 0.0
    
    return selected_features


# ==============================================================================
# ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
# ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
# ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║
# ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║██║   ██║██║╚██╗██║
# ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
#                    FONCTION PRINCIPALE DE PRÉDICTION
# ==============================================================================

def predict(audio_path: str):
    """
    Prédit si un fichier audio est Bruyant ou Normal.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PIPELINE DE PRÉDICTION                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   FICHIER AUDIO                                                         │
    │        │                                                                │
    │        ▼                                                                │
    │   ┌─────────────────┐                                                   │
    │   │ Extraction des  │   Même méthode que feature_extraction.py         │
    │   │    features     │   → 12 features sélectionnées                    │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │ Standardisation │   Avec le MÊME scaler que l'entraînement         │
    │   │   (scaler)      │   → Valeurs centrées-réduites                    │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │    Modèle       │   Réseau de neurones TensorFlow                  │
    │   │  TensorFlow     │   → Output sigmoid [0, 1]                        │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │  Interprétation │   < 0.5 → BRUYANT                                │
    │   │   du résultat   │   > 0.5 → NORMAL                                 │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │      AFFICHAGE                                                          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Paramètres:
        audio_path (str): Chemin vers le fichier audio à analyser
                          Formats supportés : .wav, .mp3, .flac, .ogg, .m4a
    
    Retourne:
        tuple: (label, confidence) où :
               - label (str) : "BRUYANT" ou "NORMAL"
               - confidence (float) : Pourcentage de confiance (0-100)
               
        None si une erreur survient
    """
    
    # ==========================================================================
    # ÉTAPE 1 : AFFICHAGE DE L'EN-TÊTE
    # ==========================================================================
    
    print_header("Prédiction Audio")
    
    # ==========================================================================
    # ÉTAPE 2 : VÉRIFICATION DES FICHIERS REQUIS
    # ==========================================================================
    # Avant de commencer, on s'assure que tous les fichiers nécessaires
    # existent. Cela évite des erreurs cryptiques plus tard.
    # ==========================================================================
    
    # --------------------------------------------------------------------------
    # Vérifier que le fichier audio existe
    # --------------------------------------------------------------------------
    if not os.path.exists(audio_path):
        print_error(f"Fichier non trouvé: {audio_path}")
        print_info("Vérifie que le chemin est correct")
        return None
    
    # --------------------------------------------------------------------------
    # Vérifier que le modèle existe
    # Si le modèle n'existe pas, c'est probablement que train_model.py
    # n'a pas encore été exécuté
    # --------------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print_error(f"Modèle non trouvé: {MODEL_PATH}")
        print_info("Lance d'abord: python train_model.py")
        return None
    
    # --------------------------------------------------------------------------
    # Vérifier que le scaler existe
    # Le scaler est sauvegardé avec le modèle par train_model.py
    # --------------------------------------------------------------------------
    if not os.path.exists(SCALER_PATH):
        print_error(f"Scaler non trouvé: {SCALER_PATH}")
        print_info("Le fichier scaler_final.pkl devrait être dans le dossier models/")
        return None
    
    # --------------------------------------------------------------------------
    # Vérifier que la liste des features existe
    # Ce fichier contient les noms des features dans l'ordre attendu
    # --------------------------------------------------------------------------
    if not os.path.exists(FEATURES_FILE):
        print_error(f"Liste des features non trouvée: {FEATURES_FILE}")
        print_info("Le fichier features_used.txt devrait être dans le dossier models/")
        return None
    
    # Afficher le fichier en cours d'analyse
    print_info(f"Fichier: {audio_path}")
    
    # ==========================================================================
    # ÉTAPE 3 : CHARGEMENT DU MODÈLE ET DU SCALER
    # ==========================================================================
    
    print_header("Chargement du modèle")
    
    # --------------------------------------------------------------------------
    # Charger la liste des features utilisées
    # 
    # Le fichier features_used.txt contient une feature par ligne :
    #   mfcc_39_mean
    #   mfcc_11_std
    #   ...
    # 
    # L'ordre est CRUCIAL car le modèle attend les features dans cet ordre !
    # --------------------------------------------------------------------------
    with open(FEATURES_FILE, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print_info(f"{len(feature_names)} features attendues")
    
    # --------------------------------------------------------------------------
    # Charger le modèle TensorFlow/Keras
    # 
    # Le fichier .keras contient tout ce qu'il faut :
    # - Architecture du réseau
    # - Poids entraînés
    # - Configuration de l'optimizer
    # --------------------------------------------------------------------------
    model = tf.keras.models.load_model(MODEL_PATH)
    print_success("Modèle chargé")
    
    # --------------------------------------------------------------------------
    # Charger le StandardScaler
    # 
    # Le scaler contient mean_ et scale_ calculés sur le dataset d'entraînement
    # On doit utiliser CES valeurs (pas en recalculer de nouvelles)
    # --------------------------------------------------------------------------
    scaler = joblib.load(SCALER_PATH)
    print_success("Scaler chargé")
    
    # ==========================================================================
    # ÉTAPE 4 : EXTRACTION DES FEATURES
    # ==========================================================================
    
    print_header("Extraction des features")
    
    print_info("Analyse du fichier audio...")
    
    try:
        # Extraire les features du fichier audio
        # On ne garde que les features listées dans feature_names
        features = extract_features(audio_path, feature_names)
        print_success(f"{len(features)} features extraites")
        
    except Exception as e:
        # En cas d'erreur (fichier corrompu, format non supporté, etc.)
        print_error(f"Erreur lors de l'extraction: {e}")
        return None
    
    # --------------------------------------------------------------------------
    # Convertir le dictionnaire en tableau numpy
    # 
    # Le modèle attend un tableau 2D de forme (n_samples, n_features)
    # Ici on a 1 seul sample, donc forme (1, n_features)
    # 
    # IMPORTANT : On itère sur feature_names pour garantir le bon ordre
    # --------------------------------------------------------------------------
    X = np.array([[features[name] for name in feature_names]])
    # X.shape = (1, 12) si on a 12 features
    
    # ==========================================================================
    # ÉTAPE 5 : STANDARDISATION
    # ==========================================================================
    
    print_header("Prédiction")
    
    # --------------------------------------------------------------------------
    # Appliquer le StandardScaler
    # 
    # Formule : X_scaled = (X - mean_train) / std_train
    # 
    # On utilise scaler.transform() (pas fit_transform !) car :
    # - fit() calcule mean et std → déjà fait sur les données d'entraînement
    # - transform() applique la transformation avec les valeurs stockées
    # 
    # C'est CRUCIAL pour que les nouvelles données soient sur la même échelle
    # que les données d'entraînement !
    # --------------------------------------------------------------------------
    X_scaled = scaler.transform(X)
    
    # ==========================================================================
    # ÉTAPE 6 : PRÉDICTION PAR LE MODÈLE
    # ==========================================================================
    
    # --------------------------------------------------------------------------
    # Faire la prédiction
    # 
    # model.predict() passe les données à travers le réseau de neurones
    # et retourne la sortie de la dernière couche (sigmoid).
    # 
    # verbose=0 supprime les messages de progression de TensorFlow
    # --------------------------------------------------------------------------
    prediction = model.predict(X_scaled, verbose=0)
    # prediction.shape = (1, 1) car 1 sample et 1 neurone de sortie
    
    # Extraire la probabilité (valeur entre 0 et 1)
    probability = prediction[0][0]
    
    # --------------------------------------------------------------------------
    # Interpréter le résultat
    # 
    # La sortie sigmoid donne une probabilité :
    # - probability proche de 0 → Classe 0 (Bruyant)
    # - probability proche de 1 → Classe 1 (Normal)
    # 
    # Le seuil de décision standard est 0.5
    # --------------------------------------------------------------------------
    
    if probability > 0.5:
        # Le modèle prédit "Normal"
        label = "NORMAL"
        confidence = probability * 100  # Confiance = probabilité
        color = Colors.GREEN            # Vert pour "OK"
        emoji = "✓"                     # Symbole positif
    else:
        # Le modèle prédit "Bruyant"
        label = "BRUYANT"
        confidence = (1 - probability) * 100  # Confiance = 1 - probabilité
        color = Colors.RED                     # Rouge pour "attention"
        emoji = "⚠"                            # Symbole d'alerte
    
    # ==========================================================================
    # ÉTAPE 7 : AFFICHAGE DU RÉSULTAT
    # ==========================================================================
    # On crée un affichage visuel attrayant avec une "boîte" colorée
    # et une barre de progression pour la confiance
    # ==========================================================================
    
    print()
    print(f"  {Colors.BOLD}Résultat:{Colors.END}")
    print()
    
    # --------------------------------------------------------------------------
    # Boîte visuelle avec le résultat
    # Utilise des caractères Unicode pour dessiner une boîte
    # --------------------------------------------------------------------------
    print(f"    {color}{Colors.BOLD}┌─────────────────────────────────┐{Colors.END}")
    print(f"    {color}{Colors.BOLD}│                                 │{Colors.END}")
    print(f"    {color}{Colors.BOLD}│   {emoji}  {label:^20}  {emoji}   │{Colors.END}")
    print(f"    {color}{Colors.BOLD}│                                 │{Colors.END}")
    print(f"    {color}{Colors.BOLD}│      Confiance: {confidence:>5.1f}%          │{Colors.END}")
    print(f"    {color}{Colors.BOLD}│                                 │{Colors.END}")
    print(f"    {color}{Colors.BOLD}└─────────────────────────────────┘{Colors.END}")
    print()
    
    # --------------------------------------------------------------------------
    # Barre de progression visuelle
    # 
    # bar_len = nombre de blocs pleins (sur 20)
    # Exemple : 85% → 17 blocs pleins, 3 blocs vides
    # --------------------------------------------------------------------------
    bar_len = int(confidence / 5)  # 100% / 5 = 20 blocs max
    bar = '█' * bar_len + '░' * (20 - bar_len)  # █ = plein, ░ = vide
    print(f"    Confiance: [{bar}] {confidence:.1f}%")
    
    # --------------------------------------------------------------------------
    # Informations techniques (en grisé)
    # Pour les utilisateurs qui veulent comprendre le score brut
    # --------------------------------------------------------------------------
    print()
    print(f"    {Colors.DIM}Score brut (sigmoid): {probability:.4f}{Colors.END}")
    print(f"    {Colors.DIM}Seuil de décision: 0.5{Colors.END}")
    
    # ==========================================================================
    # ÉTAPE 8 : AFFICHAGE DES FEATURES (BONUS)
    # ==========================================================================
    # Affiche les 5 features avec les valeurs les plus extrêmes
    # Utile pour comprendre pourquoi le modèle a pris cette décision
    # ==========================================================================
    
    print()
    print(f"  {Colors.BOLD}Features extraites (Top 5):{Colors.END}")
    
    # Trier les features par valeur absolue décroissante
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    for name, value in sorted_features:
        print(f"    • {name}: {value:.4f}")
    
    # Retourner le résultat pour une utilisation programmatique
    return label, confidence


# ==============================================================================
# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
#                    POINT D'ENTRÉE DU SCRIPT
# ==============================================================================

if __name__ == "__main__":
    """
    Point d'entrée quand le script est exécuté directement.
    
    Ce bloc ne s'exécute que si on lance :
        python predict.py
    
    Il ne s'exécute PAS si on importe le module :
        from predict import extract_features
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ARGUMENTS EN LIGNE DE COMMANDE                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  sys.argv est une liste contenant les arguments passés au script :      │
    │                                                                         │
    │  Exemple : python predict.py audio.wav                                  │
    │            sys.argv = ['predict.py', 'audio.wav']                       │
    │            sys.argv[0] = 'predict.py' (nom du script)                   │
    │            sys.argv[1] = 'audio.wav' (premier argument)                 │
    │                                                                         │
    │  Exemple : python predict.py --test                                     │
    │            sys.argv = ['predict.py', '--test']                          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    # ==========================================================================
    # VÉRIFICATION DES ARGUMENTS
    # ==========================================================================
    
    # Vérifier qu'un argument a été fourni
    # len(sys.argv) < 2 signifie qu'il n'y a que le nom du script, pas d'argument
    if len(sys.argv) < 2:
        # Afficher l'aide si aucun argument
        print(f"""
{Colors.BOLD}Predict - Prédiction sur un fichier audio{Colors.END}

{Colors.CYAN}Usage:{Colors.END}
    python predict.py <fichier_audio>
    python predict.py chemin/vers/audio.wav
    python predict.py chemin/vers/audio.mp3

{Colors.CYAN}Exemples:{Colors.END}
    python predict.py data/slices/slice_001.wav
    python predict.py ~/Musique/test.mp3
    python predict.py recording.wav

{Colors.CYAN}Options:{Colors.END}
    --test    Utilise un fichier du dataset pour tester

{Colors.CYAN}Formats supportés:{Colors.END}
    .wav, .mp3, .flac, .ogg, .m4a
    (tout format supporté par librosa/ffmpeg)

{Colors.CYAN}Prérequis:{Colors.END}
    - Avoir exécuté train_model.py au préalable
    - Les fichiers suivants doivent exister :
        • models/model_final.keras (le modèle entraîné)
        • models/scaler_final.pkl (le scaler)
        • models/features_used.txt (liste des features)
        """)
        sys.exit(1)  # Quitter avec code d'erreur
    
    # Récupérer le chemin du fichier audio (premier argument)
    audio_path = sys.argv[1]
    
    # ==========================================================================
    # MODE TEST
    # ==========================================================================
    # Si l'utilisateur tape --test, on utilise automatiquement un fichier
    # du dataset pour vérifier que tout fonctionne
    # ==========================================================================
    
    if audio_path == "--test":
        # Importer glob pour chercher des fichiers
        import glob
        
        # Chercher tous les fichiers .wav dans data/slices/
        slices = glob.glob("data/slices/*.wav")
        
        if slices:
            # Utiliser le premier fichier trouvé
            audio_path = slices[0]
            print_info(f"Mode test: {audio_path}")
        else:
            print_error("Aucun fichier dans data/slices/")
            print_info("Vérifie que tu as des fichiers audio à tester")
            sys.exit(1)
    
    # ==========================================================================
    # EXÉCUTION DE LA PRÉDICTION
    # ==========================================================================
    
    try:
        # Lancer la prédiction
        result = predict(audio_path)
        
        # Si tout s'est bien passé, result = (label, confidence)
        # Sinon, result = None
        
    except KeyboardInterrupt:
        # L'utilisateur a appuyé sur Ctrl+C pour interrompre
        print(f"\n{Colors.YELLOW}[!]{Colors.END} Annulé par l'utilisateur")
        
    except Exception as e:
        # Une erreur inattendue s'est produite
        print_error(str(e))
        # On re-raise l'exception pour voir le traceback complet (debug)
        raise
