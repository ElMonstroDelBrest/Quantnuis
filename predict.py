import os
import sys
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model  # pyright: ignore

# Configuration GPU
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ {len(gpus)} GPU(s) détecté(s)")
    try:
        # Permettre la croissance de la mémoire GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ Configuration GPU activée")
    except RuntimeError as e:
        print(f"⚠ Erreur de configuration GPU: {e}")
else:
    print("⚠ Aucun GPU détecté, utilisation du CPU")
print()

def extract_features(audio_path):
    """
    Extrait les caractéristiques mel-spectrogramme d'un fichier audio
    de la même manière que lors de l'entraînement
    """
    original_audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mels = np.mean(librosa.feature.melspectrogram(y=original_audio, sr=sample_rate).T, axis=0)
    return mels

def get_real_labels(annotation_csv='annotation.csv'):
    """
    Lit les labels réels depuis le fichier annotation.csv
    et retourne un mapping des indices du modèle vers les labels réels
    
    to_categorical crée un encodage one-hot où l'index correspond à la valeur du label.
    Si les labels sont 1,2,3,4, les indices seront 0,1,2,3,4 (mais l'index 0 n'est jamais utilisé).
    """
    if not os.path.exists(annotation_csv):
        return None, None
    
    df = pd.read_csv(annotation_csv)
    unique_labels = sorted(df['label'].unique())
    
    # Créer un mapping: index_modèle -> label_réel
    # to_categorical utilise directement la valeur du label comme index
    # Donc index 1 -> label 1, index 2 -> label 2, etc.
    label_mapping = {}
    for label in unique_labels:
        label_mapping[label] = label  # L'index du modèle = la valeur du label
    
    return label_mapping, unique_labels

def predict_audio(model_path, audio_path):
    """
    Charge le modèle et prédit le label d'un fichier audio
    
    Args:
        model_path: Chemin vers le modèle sauvegardé (model.h5)
        audio_path: Chemin vers le fichier audio à prédire
    """
    # Obtenir le mapping des labels réels
    label_mapping, real_labels = get_real_labels()
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = load_model(model_path)
    
    # Extraire les caractéristiques
    print(f"Extraction des caractéristiques de {audio_path}...")
    features = extract_features(audio_path)
    
    # Reshape pour correspondre à l'input du modèle (batch_size=1)
    features = features.reshape(1, -1)
    
    # Faire la prédiction
    print("Prédiction en cours...")
    prediction = model.predict(features, verbose=0)
    
    # Obtenir l'index du modèle avec la plus haute probabilité
    model_index = np.argmax(prediction[0])
    
    # Convertir l'index du modèle vers le label réel
    # Si label_mapping existe et que l'index correspond à un label réel
    if label_mapping and model_index in label_mapping:
        predicted_label = label_mapping[model_index]
    elif label_mapping and real_labels:
        # Si l'index 0 est prédit mais n'existe pas, prendre le deuxième meilleur
        sorted_indices = np.argsort(prediction[0])[::-1]
        for idx in sorted_indices:
            if idx in label_mapping:
                predicted_label = label_mapping[idx]
                model_index = idx
                break
    else:
        predicted_label = model_index
    
    confidence = prediction[0][model_index] * 100
    
    # Afficher les résultats
    print(f"\n{'='*50}")
    print(f"Résultat de la prédiction:")
    print(f"{'='*50}")
    print(f"Fichier audio: {audio_path}")
    print(f"Label prédit: {predicted_label}")
    print(f"Confiance: {confidence:.2f}%")
    
    if label_mapping and real_labels:
        print(f"\nProbabilités pour les classes existantes:")
        for real_label in real_labels:
            model_idx = real_label  # L'index = la valeur du label
            if model_idx < len(prediction[0]):
                prob = prediction[0][model_idx] * 100
                print(f"  Classe {real_label}: {prob:.2f}%")
    else:
        print(f"\nProbabilités pour toutes les classes:")
        for i, prob in enumerate(prediction[0]):
            print(f"  Classe {i}: {prob*100:.2f}%")
    
    print(f"{'='*50}\n")
    
    return predicted_label, confidence

if __name__ == "__main__":
    # Chemin par défaut du modèle
    model_path = "model.h5"
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"ERREUR: Le modèle {model_path} n'existe pas.")
        print("Veuillez d'abord entraîner le modèle en exécutant model.py")
        sys.exit(1)
    
    # Obtenir le chemin du fichier audio
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Utiliser le premier fichier du dossier slices par défaut
        slices_dir = "slices"
        if os.path.exists(slices_dir) and os.listdir(slices_dir):
            audio_files = sorted([f for f in os.listdir(slices_dir) if f.endswith('.wav')])
            if audio_files:
                audio_path = os.path.join(slices_dir, audio_files[0])
                print(f"Aucun fichier spécifié, utilisation de {audio_path} par défaut\n")
            else:
                print("ERREUR: Aucun fichier .wav trouvé dans le dossier 'slices'")
                sys.exit(1)
        else:
            print("ERREUR: Le dossier 'slices' n'existe pas ou est vide")
            print("Usage: python predict.py [chemin_vers_fichier_audio]")
            sys.exit(1)
    
    # Vérifier que le fichier audio existe
    if not os.path.exists(audio_path):
        print(f"ERREUR: Le fichier audio {audio_path} n'existe pas.")
        sys.exit(1)
    
    # Faire la prédiction
    predict_audio(model_path, audio_path)
