import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
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
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model_improved.h5")
MODEL_OLD_PATH = os.path.join(MODELS_DIR, "model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_MAPPING_PATH = os.path.join(MODELS_DIR, "label_mapping.pkl")
OUTPUT_DIR = "output"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "predictions_raw.csv")

# Créer les dossiers s'ils n'existent pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Détecter quelle fonction chroma est disponible
_chroma_func = None
if hasattr(librosa.feature, 'chroma_stft'):
    _chroma_func = librosa.feature.chroma_stft
elif hasattr(librosa.feature, 'chroma'):
    _chroma_func = librosa.feature.chroma

def extract_features_from_audio(audio_data, sample_rate):
    """
    Extrait les caractéristiques audio améliorées (identique à model_improved.py)
    """
    try:
        # Normaliser l'audio
        audio_data = librosa.util.normalize(audio_data)
        
        features = []
        
        # 1. Mel-spectrogramme
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        mel_mean = np.mean(mel_spec, axis=1)
        mel_std = np.std(mel_spec, axis=1)
        # S'assurer que ce sont des arrays 1D
        features.extend(np.array(mel_mean).flatten().tolist())
        features.extend(np.array(mel_std).flatten().tolist())
        
        # 2. MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        # S'assurer que ce sont des arrays 1D
        features.extend(np.array(mfcc_mean).flatten().tolist())
        features.extend(np.array(mfcc_std).flatten().tolist())
        
        # 3. Chroma features
        if _chroma_func is not None:
            try:
                chroma = _chroma_func(y=audio_data, sr=sample_rate)
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
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
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
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        features.append(zcr_mean)
        features.append(zcr_std)
        
        # 6. Tempo (rythme) - peut échouer sur de très courts segments
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
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

def seconds_to_timestamp(seconds):
    """
    Convertit un nombre de secondes en format HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def merge_consecutive_segments(predictions, max_gap_seconds=10):
    """
    Fusionne les segments consécutifs avec le même label
    si l'écart entre eux est inférieur à max_gap_seconds
    """
    if not predictions:
        return predictions
    
    def timestamp_to_seconds(timestamp):
        """Convertit HH:MM:SS en secondes"""
        parts = timestamp.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    merged = []
    current = predictions[0].copy()
    
    for next_seg in predictions[1:]:
        # Calculer l'écart entre la fin du segment actuel et le début du suivant
        current_end = timestamp_to_seconds(current['End'])
        next_start = timestamp_to_seconds(next_seg['Start'])
        gap = next_start - current_end
        
        # Si même label et écart petit, fusionner
        if (current['Label'] == next_seg['Label'] and 
            gap <= max_gap_seconds and gap >= 0):
            current['End'] = next_seg['End']
            # Prendre la fiabilité la plus élevée
            current['Reliability'] = max(current['Reliability'], next_seg['Reliability'])
        else:
            # Sauvegarder le segment actuel et commencer un nouveau
            merged.append(current)
            current = next_seg.copy()
    
    # Ajouter le dernier segment
    merged.append(current)
    
    return merged

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

def predict_segment(model, audio_segment, sample_rate, label_mapping, scaler=None):
    """
    Prédit le label d'un segment audio
    """
    # Extraire les caractéristiques
    features = extract_features_from_audio(audio_segment, sample_rate)
    
    if features is None:
        return None, 0.0
    
    # Normaliser les caractéristiques avec le scaler si disponible
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    # Faire la prédiction
    prediction = model.predict(features, verbose=0)
    
    # Obtenir l'index du modèle avec la plus haute probabilité (0 ou 1)
    model_index = np.argmax(prediction[0])
    
    # Convertir l'index du modèle vers le label réel (1=bolides, 2=autres)
    predicted_label = label_mapping.get(model_index, model_index + 1)
    
    confidence = prediction[0][model_index] * 100
    
    return predicted_label, confidence

def analyze_full_audio(model_path, audio_path, segment_duration=30, overlap=5, 
                       min_confidence=50, output_csv=OUTPUT_CSV, scaler_path=SCALER_PATH):
    """
    Analyse un fichier audio complet et génère un CSV de prédictions
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        audio_path: Chemin vers le fichier audio complet
        segment_duration: Durée de chaque segment en secondes (défaut: 30)
        overlap: Chevauchement entre segments en secondes (défaut: 5)
        min_confidence: Confiance minimale pour inclure une prédiction (défaut: 50%)
        output_csv: Nom du fichier CSV de sortie
        scaler_path: Chemin vers le fichier scaler pour normalisation (défaut: 'scaler.pkl')
    """
    # Obtenir le mapping des labels (train_to_label: 0->1, 1->2)
    label_mapping = get_label_mapping()
    
    # Charger le scaler si disponible
    scaler = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Scaler chargé depuis {scaler_path}")
        except Exception as e:
            print(f"⚠ Impossible de charger le scaler ({scaler_path}): {e}")
            print("  Les features ne seront pas normalisées.")
    else:
        print(f"⚠ Scaler non trouvé ({scaler_path}). Les features ne seront pas normalisées.")
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = load_model(model_path)
    print("Modèle chargé avec succès.\n")
    
    # Charger le fichier audio complet
    print(f"Chargement du fichier audio: {audio_path}")
    print("Cela peut prendre du temps pour un fichier de 12h...")
    audio_data, sample_rate = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
    duration_seconds = len(audio_data) / sample_rate
    print(f"Fichier audio chargé: {duration_seconds/3600:.2f} heures ({duration_seconds:.0f} secondes)\n")
    
    # Calculer les paramètres de segmentation
    segment_samples = int(segment_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step_samples = segment_samples - overlap_samples
    
    # Liste pour stocker les prédictions
    predictions = []
    
    # Parcourir le fichier audio par segments
    num_segments = int((len(audio_data) - segment_samples) / step_samples) + 1
    print(f"Analyse de {num_segments} segments (durée: {segment_duration}s, chevauchement: {overlap}s)...\n")
    
    for i in range(num_segments):
        start_sample = i * step_samples
        end_sample = start_sample + segment_samples
        
        # S'assurer qu'on ne dépasse pas la fin du fichier
        if end_sample > len(audio_data):
            end_sample = len(audio_data)
            start_sample = end_sample - segment_samples
            if start_sample < 0:
                start_sample = 0
        
        # Extraire le segment
        segment = audio_data[start_sample:end_sample]
        
        # Calculer les timestamps
        start_time_seconds = start_sample / sample_rate
        end_time_seconds = end_sample / sample_rate
        
        # Faire la prédiction
        predicted_label, confidence = predict_segment(
            model, segment, sample_rate, label_mapping, scaler
        )
        
        # Ignorer si la prédiction a échoué
        if predicted_label is None:
            continue
        
        # Convertir en timestamps
        start_timestamp = seconds_to_timestamp(start_time_seconds)
        end_timestamp = seconds_to_timestamp(end_time_seconds)
        
        # Déterminer la fiabilité basée sur la confiance
        if confidence >= 80:
            reliability = 3
        elif confidence >= 60:
            reliability = 2
        elif confidence >= min_confidence:
            reliability = 1
        else:
            # Ignorer les prédictions avec une confiance trop faible
            continue
        
        # Ajouter la prédiction
        predictions.append({
            'Start': start_timestamp,
            'End': end_timestamp,
            'Label': predicted_label,
            'Reliability': reliability
        })
        
        # Afficher la progression
        if (i + 1) % 100 == 0 or i == 0:
            progress = ((i + 1) / num_segments) * 100
            print(f"Progression: {progress:.1f}% ({i+1}/{num_segments} segments) - "
                  f"Dernière prédiction: {start_timestamp} -> {end_timestamp}, "
                  f"Label: {predicted_label}, Confiance: {confidence:.1f}%")
    
    # Fusionner les segments consécutifs avec le même label
    if predictions:
        print("Fusion des segments consécutifs...")
        predictions_merged = merge_consecutive_segments(predictions, max_gap_seconds=10)
        print(f"Segments fusionnés: {len(predictions)} -> {len(predictions_merged)}\n")
        
        # Créer le DataFrame et sauvegarder
        df_predictions = pd.DataFrame(predictions_merged)
        df_predictions.to_csv(output_csv, index=False)
        print(f"\n{'='*60}")
        print(f"Analyse terminée!")
        print(f"{len(predictions_merged)} prédictions sauvegardées dans '{output_csv}'")
        print(f"{'='*60}\n")
        
        # Afficher un résumé
        print("Résumé des prédictions:")
        label_counts = df_predictions['Label'].value_counts().sort_index()
        total = len(df_predictions)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"  Classe {label}: {count} segments ({percentage:.1f}%)")
        print()
    else:
        print("\nAucune prédiction avec une confiance suffisante n'a été trouvée.")
        print(f"Essayez de réduire min_confidence (actuellement: {min_confidence}%)")
    
    return df_predictions

if __name__ == "__main__":
    # Vérifier que le modèle existe, sinon essayer l'ancien modèle
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        model_path = MODEL_OLD_PATH
        if not os.path.exists(model_path):
            print(f"ERREUR: Aucun modèle trouvé ({MODEL_PATH} ou {MODEL_OLD_PATH}).")
            print("Veuillez d'abord entraîner le modèle en exécutant model_improved.py")
            sys.exit(1)
        else:
            print(f"⚠ Modèle amélioré non trouvé, utilisation de {model_path}")
            print("  Pour de meilleures performances, utilisez model_improved.h5")
    else:
        print(f"✓ Utilisation du modèle amélioré: {model_path}")
    
    # Obtenir le chemin du fichier audio
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Chercher automatiquement un fichier audio dans le répertoire courant
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        audio_path = None
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                # Ignorer les fichiers dans les dossiers data et output
                if not file.startswith('data') and not file.startswith('output'):
                    audio_path = file
                    break
        
        if not audio_path:
            print("ERREUR: Aucun fichier audio trouvé.")
            print("Usage: python predict_full_audio.py [chemin_vers_fichier_audio]")
            print("Ou placez un fichier audio (.wav, .mp3, .flac, etc.) dans le répertoire courant.")
            sys.exit(1)
    
    # Vérifier que le fichier audio existe
    if not os.path.exists(audio_path):
        print(f"ERREUR: Le fichier audio {audio_path} n'existe pas.")
        sys.exit(1)
    
    # Paramètres configurables (peuvent être passés en arguments si nécessaire)
    segment_duration = 30  # secondes
    overlap = 5  # secondes
    min_confidence = 50  # pourcentage
    
    # Analyser le fichier audio complet
    analyze_full_audio(
        model_path=model_path,
        audio_path=audio_path,
        segment_duration=segment_duration,
        overlap=overlap,
        min_confidence=min_confidence
    )
