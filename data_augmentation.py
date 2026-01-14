#!/usr/bin/env python3
"""
Augmentation de la base de données audio

Crée des versions modifiées des slices pour augmenter le dataset :
- Bruit gaussien
- Changement de vitesse
- Changement de pitch

Usage:
    python data_augmentation.py              # Mode interactif
    python data_augmentation.py status       # Afficher le statut
    python data_augmentation.py augment      # Augmenter tout
    python data_augmentation.py augment 1    # Augmenter seulement label 1
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# Configuration
SLICES_DIR = "data/slices"
ANNOTATION_CSV = "data/annotation.csv"
SAMPLE_RATE = 22050


def add_noise(audio: np.ndarray, factor: float = 0.005) -> np.ndarray:
    """Ajoute du bruit gaussien"""
    noise = np.random.normal(0, factor, len(audio))
    return librosa.util.normalize(audio + noise)


def stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Change la vitesse (sans changer le pitch)"""
    return librosa.util.normalize(librosa.effects.time_stretch(audio, rate=rate))


def shift_pitch(audio: np.ndarray, sr: int, steps: float) -> np.ndarray:
    """Change le pitch (sans changer la vitesse)"""
    return librosa.util.normalize(librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps))


def get_next_num() -> int:
    """Trouve le prochain numéro de slice"""
    if not os.path.exists(SLICES_DIR):
        return 1
    files = [f for f in os.listdir(SLICES_DIR) if f.startswith('slice_') and f.endswith('.wav')]
    if not files:
        return 1
    nums = [int(f.replace('slice_', '').replace('.wav', '')) for f in files]
    return max(nums) + 1


def show_status():
    """Affiche les stats de la base de données"""
    if not os.path.exists(ANNOTATION_CSV):
        print("❌ Pas de fichier d'annotations")
        return
    
    df = pd.read_csv(ANNOTATION_CSV)
    print("\n📊 Base de données:")
    print(f"   Total: {len(df)} fichiers")
    
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        print(f"   Label {label}: {count}")
    
    if 'augmentation' in df.columns:
        orig = df['augmentation'].isna().sum()
        aug = len(df) - orig
        print(f"\n   Originaux: {orig}")
        print(f"   Augmentés: {aug}")


def augment(target_label: int = None):
    """Augmente les données"""
    if not os.path.exists(ANNOTATION_CSV):
        print("❌ Pas de fichier d'annotations")
        return
    
    df = pd.read_csv(ANNOTATION_CSV)
    
    # Filtrer
    if target_label is not None:
        to_augment = df[df['label'] == target_label]
        print(f"🎯 Augmentation du label {target_label}: {len(to_augment)} fichiers")
    else:
        to_augment = df
        print(f"🎯 Augmentation de tout: {len(to_augment)} fichiers")
    
    # Ignorer les déjà augmentés
    if 'augmentation' in df.columns:
        to_augment = to_augment[to_augment['augmentation'].isna()]
        print(f"   ({len(to_augment)} originaux)")
    
    if len(to_augment) == 0:
        print("Rien à augmenter")
        return
    
    # Augmentations à appliquer
    augmentations = [
        ("noise", lambda a, sr: add_noise(a)),
        ("slow", lambda a, sr: stretch(a, 0.9)),
        ("fast", lambda a, sr: stretch(a, 1.1)),
        ("pitch_up", lambda a, sr: shift_pitch(a, sr, 2)),
        ("pitch_down", lambda a, sr: shift_pitch(a, sr, -2)),
    ]
    
    print(f"\n📈 {len(to_augment) * len(augmentations)} fichiers seront créés")
    confirm = input("Continuer? (o/n): ").strip().lower()
    if confirm not in ['o', 'y', 'oui', 'yes']:
        print("Annulé")
        return
    
    next_num = get_next_num()
    new_rows = []
    
    # Ajouter colonnes si nécessaire
    if 'source' not in df.columns:
        df['source'] = df['nfile']
    if 'augmentation' not in df.columns:
        df['augmentation'] = pd.NA
    
    for _, row in to_augment.iterrows():
        path = os.path.join(SLICES_DIR, row['nfile'])
        if not os.path.exists(path):
            continue
        
        try:
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)
            audio = librosa.util.normalize(audio)
        except Exception as e:
            print(f"⚠ Erreur {row['nfile']}: {e}")
            continue
        
        for aug_name, aug_func in augmentations:
            try:
                augmented = aug_func(audio, sr)
                new_name = f"slice_{next_num:03d}.wav"
                new_path = os.path.join(SLICES_DIR, new_name)
                
                sf.write(new_path, augmented, sr)
                
                new_rows.append({
                    'nfile': new_name,
                    'length': int(len(augmented) / sr),
                    'label': row['label'],
                    'reliability': row['reliability'],
                    'source': row['nfile'],
                    'augmentation': aug_name
                })
                
                next_num += 1
            except Exception as e:
                print(f"⚠ Erreur {aug_name} sur {row['nfile']}: {e}")
    
    # Sauvegarder
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_new], ignore_index=True)
        df_combined.to_csv(ANNOTATION_CSV, index=False)
        print(f"\n✓ {len(new_rows)} fichiers créés")
        print(f"📊 Total: {len(df_combined)} fichiers")


def main():
    if len(sys.argv) < 2:
        print("\n🔊 AUGMENTATION DE DONNÉES")
        print("=" * 30)
        print("1. Voir le statut")
        print("2. Augmenter tout")
        print("3. Augmenter un label")
        print("0. Quitter")
        
        choice = input("\nChoix: ").strip()
        
        if choice == "1":
            show_status()
        elif choice == "2":
            augment()
        elif choice == "3":
            label = input("Label (1 ou 2): ").strip()
            augment(int(label))
        elif choice == "0":
            pass
    else:
        cmd = sys.argv[1]
        if cmd == "status":
            show_status()
        elif cmd == "augment":
            label = int(sys.argv[2]) if len(sys.argv) > 2 else None
            augment(label)
        else:
            print("Usage: python data_augmentation.py [status|augment [label]]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Annulé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
