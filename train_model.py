#!/usr/bin/env python3
"""
================================================================================
                    ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION
================================================================================

Ce script entraîne un réseau de neurones (TensorFlow/Keras) pour classifier
les fichiers audio en "Bruyant" ou "Normal".

PIPELINE :
    1. Chargement des features depuis le CSV
    2. Sélection des meilleures features (Top 12 du Random Forest)
    3. Data Augmentation avec SMOTE (génère des données synthétiques)
    4. Split Train/Test (80%/20%)
    5. Standardisation des données
    6. Création du modèle (réseau de neurones)
    7. Entraînement (60 epochs)
    8. Évaluation et visualisation
    9. Sauvegarde du modèle

POURQUOI SMOTE ?
    Quand on a peu de données (ex: 44 échantillons), le modèle n'a pas assez
    d'exemples pour apprendre. SMOTE (Synthetic Minority Over-sampling Technique)
    crée de nouvelles données synthétiques en interpolant entre les points
    existants. Ça multiplie le dataset !

ARCHITECTURE DU RÉSEAU :
    Input (12 features)
        ↓
    Dense 64 neurones + ReLU + Dropout 30%
        ↓
    Dense 32 neurones + ReLU + Dropout 20%
        ↓
    Dense 1 neurone + Sigmoid → Sortie (0 ou 1)

Usage:
    python train_model.py

Fichiers générés:
    - models/model_final.keras    : Le modèle entraîné
    - models/scaler_final.pkl     : Le scaler pour normaliser les nouvelles données
    - data/training_history.png   : Courbes d'apprentissage

================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SMOTE pour l'augmentation de données
from imblearn.over_sampling import SMOTE

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ==============================================================================
# CONFIGURATION GPU (CUDA)
# ==============================================================================
# TensorFlow détecte automatiquement les GPU NVIDIA avec CUDA installé.
# On configure ici la gestion de la mémoire GPU.

def setup_gpu():
    """
    Configure TensorFlow pour utiliser le GPU CUDA si disponible.
    
    Retourne:
        str: Description du device utilisé (GPU ou CPU)
    """
    # Lister tous les GPU disponibles
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Activer la croissance dynamique de la mémoire GPU
            # Sans ça, TensorFlow réserve TOUTE la mémoire GPU dès le départ
            # Avec ça, il n'utilise que ce dont il a besoin
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Afficher les infos GPU
            gpu_info = []
            for i, gpu in enumerate(gpus):
                gpu_info.append(f"GPU {i}: {gpu.name}")
            
            return f"GPU CUDA ({len(gpus)} device(s))", gpu_info
            
        except RuntimeError as e:
            # La config doit être faite AVANT d'utiliser TensorFlow
            return f"GPU (erreur config: {e})", []
    else:
        return "CPU (pas de GPU détecté)", []


# Configurer le GPU au chargement du module
DEVICE_TYPE, GPU_INFO = setup_gpu()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FEATURES_CSV = "data/features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_final.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_final.pkl")

# ==============================================================================
# COULEURS ANSI
# ==============================================================================

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header(title: str):
    width = 50
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'─' * width}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {title.upper()}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * width}{Colors.END}")


def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")


def print_info(msg: str):
    print(f"{Colors.CYAN}[i]{Colors.END} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}[!]{Colors.END} {msg}")


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main():
    """
    Fonction principale d'entraînement du modèle.
    """
    
    # ==========================================================================
    # ÉTAPE 0 : VÉRIFICATION GPU
    # ==========================================================================
    
    print_header("Configuration Hardware")
    
    print_info(f"TensorFlow version: {tf.__version__}")
    
    if "GPU" in DEVICE_TYPE:
        print_success(f"Device: {DEVICE_TYPE}")
        for gpu in GPU_INFO:
            print(f"    {Colors.GREEN}✓{Colors.END} {gpu}")
    else:
        print_warning(f"Device: {DEVICE_TYPE}")
        print_info("Pour utiliser le GPU, installe CUDA et cuDNN")
        print(f"    {Colors.DIM}https://developer.nvidia.com/cuda-downloads{Colors.END}")
    
    # ==========================================================================
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # ==========================================================================
    
    print_header("Chargement des données")
    
    if not os.path.exists(FEATURES_CSV):
        print(f"{Colors.RED}[ERREUR]{Colors.END} Fichier non trouvé: {FEATURES_CSV}")
        print("        Lance d'abord: python feature_extraction.py")
        return
    
    df = pd.read_csv(FEATURES_CSV)
    print_info(f"{len(df)} échantillons chargés")
    
    # ==========================================================================
    # ÉTAPE 2 : SÉLECTION DES MEILLEURES FEATURES
    # ==========================================================================
    # On utilise les features identifiées comme les plus importantes par le
    # Random Forest dans feature_analysis.py
    # Moins de features = modèle plus rapide et moins de risque d'overfitting
    
    print_header("Sélection des features")
    
    # Top 12 des features les plus importantes (à ajuster selon tes résultats)
    # Tu peux les récupérer depuis data/feature_importance.csv
    top_features = [
        'mfcc_39_mean', 'mfcc_11_std', 'mfcc_37_mean', 'mfcc_21_mean',
        'mfcc_24_std', 'spectral_bandwidth_mean', 'mfcc_31_std', 'mfcc_33_mean',
        'mfcc_17_mean', 'mfcc_6_mean', 'mfcc_12_std', 'chroma_7_mean'
    ]
    
    # Charger les features depuis le fichier d'importance si disponible
    importance_file = "data/feature_importance.csv"
    if os.path.exists(importance_file):
        importance_df = pd.read_csv(importance_file)
        top_features = importance_df['Feature'].head(12).tolist()
        print_info(f"Features chargées depuis {importance_file}")
    
    # Vérifier que les colonnes existent dans le DataFrame
    available_features = [f for f in top_features if f in df.columns]
    missing_features = [f for f in top_features if f not in df.columns]
    
    if missing_features:
        print_warning(f"Features manquantes: {missing_features[:3]}...")
    
    print_success(f"{len(available_features)} features sélectionnées")
    for i, feat in enumerate(available_features, 1):
        print(f"    {i:>2}. {feat}")
    
    # Extraire X (features) et y (labels)
    X = df[available_features].values
    y = df['label'].values
    
    # Convertir les labels si nécessaire (1, 2 -> 0, 1)
    unique_labels = np.unique(y)
    print_info(f"Labels uniques: {unique_labels}")
    
    if 2 in unique_labels:
        # Mapper 1 -> 0 (Bruyant), 2 -> 1 (Normal)
        # Ou l'inverse selon ta convention
        y = np.where(y == 1, 0, 1)
        print_info("Labels convertis: 1→0 (Bruyant), 2→1 (Normal)")
    
    # ==========================================================================
    # ÉTAPE 3 : DATA AUGMENTATION AVEC SMOTE
    # ==========================================================================
    # SMOTE (Synthetic Minority Over-sampling Technique) génère des données
    # synthétiques en interpolant entre les échantillons existants.
    # C'est crucial quand on a peu de données !
    #
    # Exemple : Si on a les points A et B, SMOTE crée un point C quelque part
    # sur le segment [A, B]. Ça enrichit le dataset sans dupliquer.
    
    print_header("Data Augmentation (SMOTE)")
    
    print_info(f"Taille avant: {X.shape[0]} échantillons")
    
    # k_neighbors=3 car on a peu de données (par défaut c'est 5)
    # Plus la valeur est basse, plus on peut travailler avec peu de données
    smote = SMOTE(k_neighbors=3, random_state=42)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print_success(f"Taille après: {X_resampled.shape[0]} échantillons")
        print_info(f"Données synthétiques créées: +{X_resampled.shape[0] - X.shape[0]}")
        
        # Distribution des classes après SMOTE
        unique, counts = np.unique(y_resampled, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = "Bruyant" if label == 0 else "Normal"
            print(f"    Classe {label} ({label_name}): {count}")
    except Exception as e:
        print_warning(f"SMOTE échoué: {e}")
        print_info("Utilisation des données originales")
        X_resampled, y_resampled = X, y
    
    # ==========================================================================
    # ÉTAPE 4 : SPLIT TRAIN/TEST
    # ==========================================================================
    # On divise les données en :
    #   - Train (80%) : pour entraîner le modèle
    #   - Test (20%) : pour évaluer la performance sur des données jamais vues
    
    print_header("Split Train/Test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,      # 20% pour le test
        random_state=42,    # Reproductibilité
        stratify=y_resampled  # Garder la même proportion de classes
    )
    
    print_info(f"Train: {X_train.shape[0]} échantillons")
    print_info(f"Test:  {X_test.shape[0]} échantillons")
    
    # ==========================================================================
    # ÉTAPE 5 : STANDARDISATION
    # ==========================================================================
    # CRUCIAL pour les réseaux de neurones !
    # Les valeurs brutes des MFCC sont très différentes (-300 à +150).
    # On normalise pour avoir moyenne=0 et écart-type=1.
    #
    # IMPORTANT : On fit le scaler UNIQUEMENT sur les données d'entraînement
    # pour éviter le "data leakage" (fuite d'information du test vers le train)
    
    print_header("Standardisation")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Fit + Transform sur train
    X_test = scaler.transform(X_test)          # Transform seulement sur test
    
    print_success("Données standardisées (μ=0, σ=1)")
    
    # ==========================================================================
    # ÉTAPE 6 : CRÉATION DU MODÈLE
    # ==========================================================================
    # Architecture "Entonnoir" : on réduit progressivement le nombre de neurones
    # Input (12) → 64 → 32 → 1 (sortie)
    #
    # Techniques de régularisation pour éviter l'overfitting :
    #   - L2 regularization : pénalise les gros poids (force la simplicité)
    #   - Dropout : éteint aléatoirement des neurones pendant l'entraînement
    #               (force le réseau à ne pas dépendre d'un seul neurone)
    
    print_header("Création du modèle")
    
    model = models.Sequential([
        # Couche d'entrée
        layers.Input(shape=(X_train.shape[1],)),
        
        # Première couche cachée : 64 neurones
        # ReLU = max(0, x) : fonction d'activation non-linéaire
        # L2(0.001) = pénalité sur les poids pour éviter l'overfitting
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),  # Éteint 30% des neurones aléatoirement
        
        # Deuxième couche cachée : 32 neurones
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),  # Éteint 20% des neurones
        
        # Couche de sortie : 1 neurone avec Sigmoid
        # Sigmoid : transforme la sortie en probabilité [0, 1]
        # Si sortie > 0.5 → classe 1, sinon → classe 0
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilation : définit comment le modèle apprend
    # Optimizer Adam : algorithme d'optimisation adaptatif (le plus utilisé)
    # Loss binary_crossentropy : fonction de coût pour classification binaire
    # Metrics accuracy : ce qu'on veut mesurer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Afficher l'architecture
    print_info("Architecture du réseau:")
    model.summary()
    
    # ==========================================================================
    # ÉTAPE 7 : ENTRAÎNEMENT
    # ==========================================================================
    # Le modèle va voir les données plusieurs fois (epochs).
    # À chaque epoch, il ajuste ses poids pour réduire l'erreur.
    #
    # batch_size=16 : on traite 16 échantillons à la fois
    # validation_data : on évalue sur le test set à chaque epoch
    
    print_header("Entraînement")
    
    print_info("Démarrage de l'entraînement (60 epochs)...")
    print()
    
    # Callback pour afficher la progression
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                acc = logs.get('accuracy', 0) * 100
                val_acc = logs.get('val_accuracy', 0) * 100
                loss = logs.get('loss', 0)
                print(f"    Epoch {epoch+1:>2}/60 - Accuracy: {acc:.1f}% - Val Accuracy: {val_acc:.1f}% - Loss: {loss:.4f}")
    
    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0,  # Pas d'affichage par défaut
        callbacks=[ProgressCallback()]
    )
    
    # ==========================================================================
    # ÉTAPE 8 : ÉVALUATION
    # ==========================================================================
    
    print_header("Résultats")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print()
    print(f"  {Colors.BOLD}Performance finale:{Colors.END}")
    print(f"    • Précision (Accuracy): {Colors.GREEN}{accuracy*100:.2f}%{Colors.END}")
    print(f"    • Erreur (Loss):        {loss:.4f}")
    
    # Interprétation
    if accuracy >= 0.9:
        print(f"\n  {Colors.GREEN}✓ Excellent ! Le modèle est très performant.{Colors.END}")
    elif accuracy >= 0.75:
        print(f"\n  {Colors.YELLOW}⚠ Bon résultat, mais peut être amélioré.{Colors.END}")
    else:
        print(f"\n  {Colors.RED}✗ Le modèle a besoin de plus de données ou d'ajustements.{Colors.END}")
    
    # ==========================================================================
    # ÉTAPE 9 : VISUALISATION
    # ==========================================================================
    
    print_header("Visualisation")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1 : Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#2a9d8f')
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#e63946')
    axes[0].set_title('Précision du Modèle', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Graphique 2 : Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, color='#2a9d8f')
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#e63946')
    axes[1].set_title('Erreur (Loss)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/training_history.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print_success("Courbes sauvegardées: data/training_history.png")
    plt.show()
    
    # ==========================================================================
    # ÉTAPE 10 : SAUVEGARDE
    # ==========================================================================
    
    print_header("Sauvegarde")
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Sauvegarder le modèle Keras
    model.save(MODEL_PATH)
    print_success(f"Modèle sauvegardé: {MODEL_PATH}")
    
    # Sauvegarder le scaler (nécessaire pour prédire sur de nouvelles données)
    joblib.dump(scaler, SCALER_PATH)
    print_success(f"Scaler sauvegardé: {SCALER_PATH}")
    
    # Sauvegarder la liste des features utilisées
    features_file = os.path.join(MODEL_DIR, "features_used.txt")
    with open(features_file, 'w') as f:
        f.write('\n'.join(available_features))
    print_success(f"Features sauvegardées: {features_file}")
    
    # ==========================================================================
    # RÉSUMÉ FINAL
    # ==========================================================================
    
    print_header("Résumé")
    
    print(f"""
  {Colors.BOLD}Entraînement terminé !{Colors.END}
  
  • Données originales:    {X.shape[0]} échantillons
  • Après SMOTE:           {X_resampled.shape[0]} échantillons
  • Features utilisées:    {len(available_features)}
  • Précision finale:      {Colors.GREEN}{accuracy*100:.2f}%{Colors.END}
  
  {Colors.BOLD}Fichiers générés:{Colors.END}
  • {MODEL_PATH}
  • {SCALER_PATH}
  • {features_file}
  • data/training_history.png
  
  {Colors.BOLD}Pour utiliser le modèle:{Colors.END}
  
    model = tf.keras.models.load_model('{MODEL_PATH}')
    scaler = joblib.load('{SCALER_PATH}')
    
    # Prédire sur de nouvelles données
    X_new = scaler.transform(new_features)
    prediction = model.predict(X_new)
    
  {Colors.GREEN}✓{Colors.END} Prêt pour la production !
    """)


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[!]{Colors.END} Annulé")
    except Exception as e:
        print(f"{Colors.RED}[ERREUR]{Colors.END} {e}")
        raise
