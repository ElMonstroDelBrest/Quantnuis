import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
import pickle
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

# Configuration des chemins
DATA_DIR = "data"
ANNOTATION_CSV = os.path.join(DATA_DIR, "annotation.csv")
SLICES_DIR = os.path.join(DATA_DIR, "slices")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model_improved.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_MAPPING_PATH = os.path.join(MODELS_DIR, "label_mapping.pkl")

# Détecter quelle fonction chroma est disponible
_chroma_func = None
if hasattr(librosa.feature, 'chroma_stft'):
    _chroma_func = librosa.feature.chroma_stft
elif hasattr(librosa.feature, 'chroma'):
    _chroma_func = librosa.feature.chroma

def extract_enhanced_features(audio_path, sample_rate=22050):
    """
    Extrait des caractéristiques audio améliorées (identique à model_improved.py)
    """
    try:
        original_audio, sr = librosa.load(audio_path, sr=sample_rate, res_type='kaiser_fast')
        original_audio = librosa.util.normalize(original_audio)
        
        features = []
        
        # 1. Mel-spectrogramme
        mel_spec = librosa.feature.melspectrogram(y=original_audio, sr=sr, n_mels=128)
        mel_mean = np.mean(mel_spec, axis=1)
        mel_std = np.std(mel_spec, axis=1)
        # S'assurer que ce sont des arrays 1D
        features.extend(np.array(mel_mean).flatten().tolist())
        features.extend(np.array(mel_std).flatten().tolist())
        
        # 2. MFCC
        mfccs = librosa.feature.mfcc(y=original_audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        # S'assurer que ce sont des arrays 1D
        features.extend(np.array(mfcc_mean).flatten().tolist())
        features.extend(np.array(mfcc_std).flatten().tolist())
        
        # 3. Chroma features
        if _chroma_func is not None:
            try:
                chroma = _chroma_func(y=original_audio, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                # S'assurer que c'est un array 1D de taille 12
                chroma_array = np.array(chroma_mean).flatten()
                if len(chroma_array) != 12:
                    # Si la taille n'est pas 12, prendre les 12 premiers ou pad avec des zéros
                    if len(chroma_array) > 12:
                        chroma_array = chroma_array[:12]
                    else:
                        chroma_array = np.pad(chroma_array, (0, 12 - len(chroma_array)), 'constant')
                features.extend(chroma_array.tolist())
            except Exception:
                # Si l'extraction échoue, utiliser des zéros
                features.extend([0.0] * 12)
        else:
            # Si chroma n'est pas disponible, utiliser des zéros pour maintenir la dimension
            features.extend([0.0] * 12)
        
        # 4. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=original_audio, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        # S'assurer que c'est un array 1D de taille fixe (généralement 7)
        contrast_array = np.array(contrast_mean).flatten()
        # Spectral contrast retourne généralement 7 valeurs
        if len(contrast_array) != 7:
            if len(contrast_array) > 7:
                contrast_array = contrast_array[:7]
            else:
                contrast_array = np.pad(contrast_array, (0, 7 - len(contrast_array)), 'constant')
        features.extend(contrast_array.tolist())
        
        # 5. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(original_audio)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        features.append(zcr_mean)
        features.append(zcr_std)
        
        # 6. Tempo (rythme) - peut échouer sur de très courts segments
        try:
            tempo, _ = librosa.beat.beat_track(y=original_audio, sr=sr)
            # Extraire la valeur scalaire avant conversion
            tempo_value = float(np.asarray(tempo).item()) if np.ndim(tempo) > 0 else float(tempo)
            features.append(tempo_value)
        except:
            # Si le tempo ne peut pas être calculé, utiliser une valeur par défaut
            features.append(120.0)  # BPM par défaut
        
        # Convertir en array numpy et s'assurer que c'est 1D
        features_array = np.array(features, dtype=np.float32)
        return features_array.flatten()
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {e}")
        return None

def get_label_mapping():
    """
    Charge le mapping des labels depuis le fichier sauvegardé
    Retourne le mapping train_to_label (0->1, 1->2)
    """
    if os.path.exists(LABEL_MAPPING_PATH):
        with open(LABEL_MAPPING_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # Mapping par défaut si le fichier n'existe pas
        return {0: 1, 1: 2}

def predict_audio(model_path, audio_path, scaler_path=SCALER_PATH):
    """
    Charge le modèle amélioré et prédit le label d'un fichier audio
    """
    # Charger le scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler chargé depuis {scaler_path}")
    else:
        print(f"ATTENTION: Scaler non trouvé ({scaler_path}). Utilisation sans normalisation.")
        scaler = None
    
    # Obtenir le mapping des labels (train_to_label: 0->1, 1->2)
    label_mapping = get_label_mapping()
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = load_model(model_path)
    print("✓ Modèle chargé avec succès")
    
    # Extraire les caractéristiques améliorées
    print(f"Extraction des caractéristiques de {audio_path}...")
    features = extract_enhanced_features(audio_path)
    
    if features is None:
        print("ERREUR: Impossible d'extraire les caractéristiques")
        return None, None
    
    # Normaliser les caractéristiques
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    # Faire la prédiction
    print("Prédiction en cours...")
    prediction = model.predict(features, verbose=0)
    
    # Obtenir l'index du modèle avec la plus haute probabilité (0 ou 1)
    model_index = np.argmax(prediction[0])
    
    # Convertir l'index du modèle vers le label réel (1=bolides, 2=autres)
    predicted_label = label_mapping.get(model_index, model_index + 1)
    
    confidence = prediction[0][model_index] * 100
    
    # Afficher les résultats
    print(f"\n{'='*50}")
    print(f"Résultat de la prédiction:")
    print(f"{'='*50}")
    print(f"Fichier audio: {audio_path}")
    label_name = "bolides" if predicted_label == 1 else "autres"
    print(f"Label prédit: {predicted_label} ({label_name})")
    print(f"Confiance: {confidence:.2f}%")
    
    print(f"\nProbabilités pour les classes:")
    for train_idx in [0, 1]:
        real_label = label_mapping.get(train_idx, train_idx + 1)
        label_name = "bolides" if real_label == 1 else "autres"
        prob = prediction[0][train_idx] * 100
        print(f"  Classe {real_label} ({label_name}): {prob:.2f}%")
    
    print(f"{'='*50}\n")
    
    return predicted_label, confidence

if __name__ == "__main__":
    # Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR: Le modèle {MODEL_PATH} n'existe pas.")
        print("Veuillez d'abord entraîner le modèle amélioré en exécutant model_improved.py")
        sys.exit(1)
    
    # Obtenir le chemin du fichier audio
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Utiliser le premier fichier du dossier slices par défaut
        if os.path.exists(SLICES_DIR) and os.listdir(SLICES_DIR):
            audio_files = sorted([f for f in os.listdir(SLICES_DIR) if f.endswith('.wav')])
            if audio_files:
                audio_path = os.path.join(SLICES_DIR, audio_files[0])
                print(f"Aucun fichier spécifié, utilisation de {audio_path} par défaut\n")
            else:
                print(f"ERREUR: Aucun fichier .wav trouvé dans le dossier '{SLICES_DIR}'")
                sys.exit(1)
        else:
            print(f"ERREUR: Le dossier '{SLICES_DIR}' n'existe pas ou est vide")
            print("Usage: python predict_improved.py [chemin_vers_fichier_audio]")
            sys.exit(1)
    
    # Vérifier que le fichier audio existe
    if not os.path.exists(audio_path):
        print(f"ERREUR: Le fichier audio {audio_path} n'existe pas.")
        sys.exit(1)
    
    # Faire la prédiction
    predict_audio(MODEL_PATH, audio_path)
