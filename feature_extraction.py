#!/usr/bin/env python3
"""
Extraction massive de caractéristiques audio

Extrait ~100 caractéristiques par fichier audio :
- RMS, ZCR (volume, passages par zéro)
- Spectral features (centroid, bandwidth, rolloff, flatness, contrast)
- Harmoniques vs Percussifs
- 40 MFCCs (timbre)
- Chroma (tonalité)

Usage:
    python feature_extraction.py              # Extraire tout
    python feature_extraction.py status       # Voir si features existent
    python feature_extraction.py --label 1    # Extraire seulement label 1
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings('ignore')

# Configuration
SLICES_DIR = "data/slices"
ANNOTATION_CSV = "data/annotation.csv"
FEATURES_CSV = "data/features.csv"


def extract_massive_features(file_path: str) -> dict | None:
    """
    Extrait ~100 caractéristiques audio d'un fichier
    
    Returns:
        dict avec toutes les features, ou None si erreur
    """
    try:
        # Chargement
        y, sr = librosa.load(file_path, res_type='kaiser_fast')
        
        # Fichier vide ou corrompu
        if len(y) == 0:
            return None

        features = {}

        # --- 1. FONCTIONS DE BASE ---
        # RMS (Volume)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # --- 2. SPECTRAL FEATURES ---
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(cent))
        features['spectral_centroid_std'] = float(np.std(cent))

        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(bw))
        features['spectral_bandwidth_std'] = float(np.std(bw))

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        features['spectral_rolloff_std'] = float(np.std(rolloff))
        
        # Spectral Flatness (ressemblance au bruit blanc)
        flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness_mean'] = float(np.mean(flatness))
        features['spectral_flatness_std'] = float(np.std(flatness))
        
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(contrast))
        features['spectral_contrast_std'] = float(np.std(contrast))

        # --- 3. HARMONIC & PERCUSSIVE ---
        y_harm, y_perc = librosa.effects.hpss(y)
        features['harm_mean'] = float(np.mean(np.abs(y_harm)))
        features['harm_std'] = float(np.std(y_harm))
        features['perc_mean'] = float(np.mean(np.abs(y_perc)))
        features['perc_std'] = float(np.std(y_perc))
        # Ratio harmonique/percussif
        features['harm_perc_ratio'] = float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-10))

        # --- 4. MFCCs (40 coefficients) ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        for i in range(40):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
        # --- 5. CHROMA ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        # Chroma par note (12 notes)
        for i in range(12):
            features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))

        # --- 6. TEMPO ---
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(np.asarray(tempo).item()) if np.ndim(tempo) > 0 else float(tempo)
        except:
            features['tempo'] = 0.0

        # --- 7. STATISTIQUES GLOBALES ---
        features['duration'] = float(len(y) / sr)
        features['energy'] = float(np.sum(y**2))
        features['max_amplitude'] = float(np.max(np.abs(y)))

        return features

    except Exception as e:
        print(f"  ❌ Erreur sur {file_path}: {e}")
        return None


def extract_all(target_label: int = None, force: bool = False):
    """
    Extrait les features de tous les slices
    
    Args:
        target_label: Si spécifié, n'extrait que ce label
        force: Si True, réextrait même si déjà fait
    """
    print("=" * 50)
    print("EXTRACTION DES CARACTÉRISTIQUES")
    print("=" * 50)
    
    if not os.path.exists(ANNOTATION_CSV):
        print("❌ Pas de fichier d'annotations")
        return
    
    # Charger les annotations
    df_ann = pd.read_csv(ANNOTATION_CSV)
    
    # Charger les features existantes
    df_existing = None
    existing_files = set()
    if os.path.exists(FEATURES_CSV) and not force:
        df_existing = pd.read_csv(FEATURES_CSV)
        existing_files = set(df_existing['nfile'].values)
        print(f"📄 {len(existing_files)} features déjà extraites")
    
    # Filtrer par label
    if target_label is not None:
        df_ann = df_ann[df_ann['label'] == target_label]
        print(f"🎯 Label {target_label}: {len(df_ann)} fichiers")
    else:
        print(f"📊 {len(df_ann)} fichiers à traiter")
    
    # Filtrer les déjà traités
    if not force:
        df_to_process = df_ann[~df_ann['nfile'].isin(existing_files)]
        print(f"📥 {len(df_to_process)} nouveaux fichiers à extraire")
    else:
        df_to_process = df_ann
        print(f"🔄 Réextraction forcée de {len(df_to_process)} fichiers")
    
    if len(df_to_process) == 0:
        print("✓ Tout est déjà extrait")
        return
    
    # Extraction
    new_rows = []
    for i, row in df_to_process.iterrows():
        path = os.path.join(SLICES_DIR, row['nfile'])
        
        if not os.path.exists(path):
            print(f"  ⚠ {row['nfile']} non trouvé")
            continue
        
        features = extract_massive_features(path)
        
        if features:
            features['nfile'] = row['nfile']
            features['label'] = row['label']
            features['reliability'] = row['reliability']
            new_rows.append(features)
            
            if len(new_rows) % 20 == 0:
                print(f"  ✓ {len(new_rows)} fichiers traités...")
    
    if not new_rows:
        print("⚠ Aucune feature extraite")
        return
    
    # Fusionner avec les existants
    df_new = pd.DataFrame(new_rows)
    
    if df_existing is not None and not force:
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    
    # Supprimer doublons
    df_final = df_final.drop_duplicates(subset=['nfile'], keep='last')
    
    # Sauvegarder
    df_final.to_csv(FEATURES_CSV, index=False)
    
    print(f"\n✓ {len(new_rows)} fichiers extraits")
    print(f"📊 Total: {len(df_final)} fichiers avec features")
    print(f"📋 {len(df_final.columns)} caractéristiques par fichier")
    print(f"💾 Sauvegardé: {FEATURES_CSV}")


def show_status():
    """Affiche le statut des features"""
    print("=" * 50)
    print("STATUT DES FEATURES")
    print("=" * 50)
    
    if not os.path.exists(FEATURES_CSV):
        print("\n❌ Pas de fichier de features")
        print(f"   Lancez: python feature_extraction.py")
        return
    
    df = pd.read_csv(FEATURES_CSV)
    
    print(f"\n📊 {len(df)} fichiers avec features")
    print(f"📋 {len(df.columns) - 3} caractéristiques extraites")
    
    # Distribution par label
    print("\n📈 Distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        print(f"    Label {label}: {count}")
    
    # Colonnes
    feature_cols = [c for c in df.columns if c not in ['nfile', 'label', 'reliability']]
    print(f"\n📋 Catégories de features:")
    print(f"    - Base (RMS, ZCR): 4")
    print(f"    - Spectral: 10")
    print(f"    - Harmonic/Percussive: 5")
    print(f"    - MFCCs: 80 (40 × 2)")
    print(f"    - Chroma: 14")
    print(f"    - Autres: {len(feature_cols) - 113}")


def main():
    if len(sys.argv) < 2:
        print("\n🎵 EXTRACTION DE CARACTÉRISTIQUES")
        print("=" * 35)
        print("1. Extraire toutes les features")
        print("2. Extraire un label spécifique")
        print("3. Réextraire tout (force)")
        print("4. Voir le statut")
        print("0. Quitter")
        
        choice = input("\nChoix: ").strip()
        
        if choice == "1":
            extract_all()
        elif choice == "2":
            label = input("Label: ").strip()
            extract_all(target_label=int(label))
        elif choice == "3":
            extract_all(force=True)
        elif choice == "4":
            show_status()
    else:
        cmd = sys.argv[1]
        
        if cmd == "status":
            show_status()
        elif cmd == "--label":
            label = int(sys.argv[2]) if len(sys.argv) > 2 else None
            extract_all(target_label=label)
        elif cmd == "--force":
            extract_all(force=True)
        else:
            extract_all()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Annulé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
