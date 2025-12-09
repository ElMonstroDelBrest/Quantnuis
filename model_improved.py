import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical  # pyright: ignore
from tensorflow.keras.models import Sequential  # pyright: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization  # pyright: ignore
from tensorflow.keras.optimizers import Adam  # pyright: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # pyright: ignore
import warnings
warnings.filterwarnings('ignore')

# Détecter quelle fonction chroma est disponible
_chroma_func = None
if hasattr(librosa.feature, 'chroma_stft'):
    _chroma_func = librosa.feature.chroma_stft
elif hasattr(librosa.feature, 'chroma'):
    _chroma_func = librosa.feature.chroma
else:
    print("ATTENTION: Aucune fonction chroma disponible dans librosa.feature")
    print("Les features chroma seront remplacées par des zéros")

def extract_enhanced_features(audio_path, sample_rate=22050):
    """
    Extrait des caractéristiques audio améliorées:
    - Mel-spectrogramme complet (pas seulement la moyenne)
    - MFCC
    - Chroma
    - Spectral contrast
    - Zero crossing rate
    """
    try:
        original_audio, sr = librosa.load(audio_path, sr=sample_rate, res_type='kaiser_fast')
        
        # Normaliser l'audio
        original_audio = librosa.util.normalize(original_audio)
        
        features = []
        
        # 1. Mel-spectrogramme (on garde la forme temporelle)
        mel_spec = librosa.feature.melspectrogram(y=original_audio, sr=sr, n_mels=128)
        # Prendre la moyenne ET l'écart-type pour capturer plus d'information
        mel_mean = np.mean(mel_spec, axis=1)
        mel_std = np.std(mel_spec, axis=1)
        # S'assurer que ce sont des arrays 1D
        features.extend(np.array(mel_mean).flatten().tolist())
        features.extend(np.array(mel_std).flatten().tolist())
        
        # 2. MFCC (Mel-frequency cepstral coefficients)
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
            except Exception as e:
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
            features.append(float(tempo))
        except:
            # Si le tempo ne peut pas être calculé, utiliser une valeur par défaut
            features.append(120.0)  # BPM par défaut
        
        # Convertir en array numpy et s'assurer que c'est 1D
        features_array = np.array(features, dtype=np.float32)
        return features_array.flatten()
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de {audio_path}: {e}")
        return None

def augment_audio(audio, sample_rate):
    """
    Augmente les données audio pour créer plus d'échantillons d'entraînement
    """
    augmented = []
    
    # 1. Original
    augmented.append(audio)
    
    # 2. Ajout de bruit gaussien léger
    noise = np.random.normal(0, 0.01, len(audio))
    augmented.append(audio + noise)
    
    # 3. Changement de pitch (time stretching)
    try:
        augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    except:
        pass
    
    # 4. Changement de vitesse (pitch shifting)
    try:
        augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2))
    except:
        pass
    
    return augmented

print("="*60)
print("MODÈLE AMÉLIORÉ - EXTRACTION DES CARACTÉRISTIQUES")
print("="*60)

features = []
labels = []

df = pd.read_csv('annotation.csv')
print(f"\nChargement de {len(df)} fichiers audio...")

for i in range(len(df)):
    audio_path = os.path.join('slices', str(df['nfile'].values[i]))
    
    if not os.path.exists(audio_path):
        print(f"ATTENTION: Fichier non trouvé: {audio_path}")
        continue
    
    # Extraire les caractéristiques améliorées
    feat = extract_enhanced_features(audio_path)
    if feat is not None:
        features.append(feat)
        labels.append(df['label'].values[i])
        
        if (i + 1) % 10 == 0:
            print(f"  Traité {i+1}/{len(df)} fichiers...")

print(f"\n✓ {len(features)} fichiers traités avec succès")

if len(features) == 0:
    print("\nERREUR: Aucune caractéristique n'a pu être extraite!")
    print("Vérifiez que les fichiers audio existent et que librosa est correctement installé.")
    import sys
    sys.exit(1)

print(f"  Dimension des caractéristiques: {len(features[0])}")

# Convertir en arrays numpy
X = np.array(features)
Y = np.array(labels)

# Normaliser les caractéristiques (très important!)
print("\nNormalisation des caractéristiques...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Sauvegarder le scaler pour l'utiliser lors de la prédiction
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler sauvegardé dans 'scaler.pkl'")

# Encoder les labels en one-hot
Y_categorical = to_categorical(Y)
num_classes = Y_categorical.shape[1]

print(f"\nNombre de classes: {num_classes}")
print(f"Distribution des labels: {np.bincount(Y)}")

# Séparation train/test
# Vérifier si on peut utiliser la stratification
# (nécessite au moins 2 membres par classe)
min_class_count = np.min(np.bincount(Y))
if min_class_count >= 2:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_categorical, 
        test_size=0.2, 
        random_state=42,
        stratify=Y  # Assure une distribution équilibrée
    )
    print("✓ Stratification activée (distribution équilibrée)")
else:
    # Si une classe a moins de 2 membres, on ne peut pas stratifier
    print(f"⚠ Stratification désactivée (classe minimale: {min_class_count} membres)")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_categorical, 
        test_size=0.2, 
        random_state=42
    )

print(f"\nDonnées d'entraînement: {len(X_train)} échantillons")
print(f"Données de test: {len(X_test)} échantillons")

# AUGMENTATION DES DONNÉES (très important avec peu de données!)
print("\n" + "="*60)
print("AUGMENTATION DES DONNÉES")
print("="*60)

# Pour chaque fichier d'entraînement, on va charger l'audio et l'augmenter
X_train_augmented = []
Y_train_augmented = []

# On doit mapper les indices d'entraînement aux fichiers originaux
# Utiliser la même logique de stratification que pour X_train/X_test
if min_class_count >= 2:
    train_indices = train_test_split(
        np.arange(len(X)), Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )[0]  # On prend seulement les indices d'entraînement
else:
    train_indices = train_test_split(
        np.arange(len(X)), Y,
        test_size=0.2,
        random_state=42
    )[0]  # On prend seulement les indices d'entraînement

print(f"Augmentation des {len(train_indices)} échantillons d'entraînement...")
temp_path = 'temp_aug.wav'

for idx in train_indices:
    audio_path = os.path.join('slices', str(df['nfile'].values[idx]))
    label = df['label'].values[idx]
    
    try:
        audio, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
        audio = librosa.util.normalize(audio)
        
        # Créer des versions augmentées (on ne ré-ajoute pas l'original, il est déjà dans X_train)
        # 1. Bruit gaussien
        try:
            noise = np.random.normal(0, 0.005, len(audio))
            audio_noisy = librosa.util.normalize(audio + noise)
            sf.write(temp_path, audio_noisy, sr)
            feat = extract_enhanced_features(temp_path, sr)
            if feat is not None:
                X_train_augmented.append(feat)
                Y_train_augmented.append(label)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # 2. Time stretch (ralentir)
        try:
            audio_stretched = librosa.effects.time_stretch(audio, rate=0.9)
            audio_stretched = librosa.util.normalize(audio_stretched)
            sf.write(temp_path, audio_stretched, sr)
            feat = extract_enhanced_features(temp_path, sr)
            if feat is not None:
                X_train_augmented.append(feat)
                Y_train_augmented.append(label)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # 3. Time stretch (accélérer)
        try:
            audio_stretched = librosa.effects.time_stretch(audio, rate=1.1)
            audio_stretched = librosa.util.normalize(audio_stretched)
            sf.write(temp_path, audio_stretched, sr)
            feat = extract_enhanced_features(temp_path, sr)
            if feat is not None:
                X_train_augmented.append(feat)
                Y_train_augmented.append(label)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # 4. Pitch shift (monter)
        try:
            audio_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            audio_shifted = librosa.util.normalize(audio_shifted)
            sf.write(temp_path, audio_shifted, sr)
            feat = extract_enhanced_features(temp_path, sr)
            if feat is not None:
                X_train_augmented.append(feat)
                Y_train_augmented.append(label)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if (len(X_train_augmented) // len(train_indices)) % 10 == 0 and len(X_train_augmented) > 0:
            print(f"  {len(X_train_augmented)} échantillons augmentés créés...")
        
    except Exception as e:
        print(f"Erreur lors de l'augmentation de {audio_path}: {e}")
        continue

# Nettoyer le fichier temporaire s'il existe encore
if os.path.exists(temp_path):
    os.remove(temp_path)

# Convertir les données augmentées
if len(X_train_augmented) > 0:
    X_train_augmented = np.array(X_train_augmented)
    Y_train_augmented = np.array(Y_train_augmented)
    
    # Normaliser les données augmentées avec le même scaler
    X_train_augmented = scaler.transform(X_train_augmented)
    
    # Encoder les labels augmentés avec le même nombre de classes que Y_train
    # to_categorical utilise le max+1 comme nombre de classes, donc on doit s'assurer
    # que toutes les classes sont représentées
    Y_train_augmented_categorical = to_categorical(Y_train_augmented, num_classes=num_classes)
    
    # Combiner avec les données originales
    X_train_final = np.vstack([X_train, X_train_augmented])
    Y_train_final = np.vstack([Y_train, Y_train_augmented_categorical])
    
    print(f"✓ Données augmentées: {len(X_train)} -> {len(X_train_final)} échantillons")
else:
    print("⚠ Aucune donnée augmentée créée, utilisation des données originales uniquement")
    X_train_final = X_train
    Y_train_final = Y_train

# ARCHITECTURE AMÉLIORÉE DU MODÈLE
print("\n" + "="*60)
print("CRÉATION DU MODÈLE")
print("="*60)

# Architecture avec BatchNormalization et Dropout pour éviter le surapprentissage
model = Sequential([
    # Première couche avec normalisation
    Dense(512, input_dim=X_train_final.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # Deuxième couche
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    # Troisième couche
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Quatrième couche
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Couche de sortie
    Dense(num_classes, activation='softmax')
])

# Compiler avec un learning rate plus bas et decay
optimizer = Adam(learning_rate=0.0001, decay=1e-6)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

print("\nArchitecture du modèle:")
model.summary()

# CALLBACKS pour améliorer l'entraînement
callbacks = [
    # Arrêt anticipé si pas d'amélioration
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Sauvegarder le meilleur modèle
    ModelCheckpoint(
        'model_improved_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # Réduire le learning rate si plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
]

# ENTRAÎNEMENT
print("\n" + "="*60)
print("ENTRAÎNEMENT DU MODÈLE")
print("="*60)

history = model.fit(
    X_train_final, Y_train_final,
    epochs=200,  # Plus d'epochs car on a early stopping
    batch_size=16,  # Batch size plus petit pour plus de stabilité
    validation_data=(X_test, Y_test),
    callbacks=callbacks,
    verbose=1
)

# Sauvegarder le modèle final
model.save('model_improved.h5')
print("\n✓ Modèle sauvegardé dans 'model_improved.h5'")

# ÉVALUATION
print("\n" + "="*60)
print("ÉVALUATION DU MODÈLE")
print("="*60)

train_loss, train_acc, train_topk = model.evaluate(X_train_final, Y_train_final, verbose=0)
test_loss, test_acc, test_topk = model.evaluate(X_test, Y_test, verbose=0)

print(f"\nRésultats sur les données d'entraînement:")
print(f"  Loss: {train_loss:.4f}")
print(f"  Accuracy: {train_acc*100:.2f}%")
print(f"  Top-K Accuracy: {train_topk*100:.2f}%")

print(f"\nRésultats sur les données de test:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc*100:.2f}%")
print(f"  Top-K Accuracy: {test_topk*100:.2f}%")

# Graphiques de l'historique d'entraînement
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Évolution de la Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Évolution de l\'Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_improved.png', dpi=150)
print("\n✓ Graphiques sauvegardés dans 'training_history_improved.png'")

print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ!")
print("="*60)
