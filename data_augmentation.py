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

# Couleurs ANSI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def print_header(title: str):
    """Affiche un titre formaté"""
    width = 40
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'─' * width}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {title.upper()}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * width}{Colors.END}")

def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}[ERREUR]{Colors.END} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}[!]{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}[i]{Colors.END} {msg}")

def print_stat(label: str, value, indent: int = 0):
    """Affiche une statistique alignée"""
    spaces = "  " * indent
    print(f"{spaces}{Colors.DIM}{label:<20}{Colors.END} {Colors.BOLD}{value}{Colors.END}")


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
        print_error("Pas de fichier d'annotations")
        return
    
    df = pd.read_csv(ANNOTATION_CSV)
    
    print_header("Base de données")
    print_stat("Total fichiers", len(df))
    print()
    
    print(f"  {Colors.BOLD}Distribution des labels:{Colors.END}")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        pct = count / len(df) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    Label {label}:  {bar} {count:>4} ({pct:.1f}%)")
    
    if 'augmentation' in df.columns:
        orig = df['augmentation'].isna().sum()
        aug = len(df) - orig
        print()
        print(f"  {Colors.BOLD}Composition:{Colors.END}")
        print(f"    Originaux:  {orig}")
        print(f"    Augmentés:  {aug}")


def augment(target_label: int = None):
    """Augmente les données"""
    if not os.path.exists(ANNOTATION_CSV):
        print_error("Pas de fichier d'annotations")
        return
    
    df = pd.read_csv(ANNOTATION_CSV)
    
    print_header("Augmentation")
    
    # Filtrer
    if target_label is not None:
        to_augment = df[df['label'] == target_label]
        print_info(f"Cible: label {target_label} ({len(to_augment)} fichiers)")
    else:
        to_augment = df
        print_info(f"Cible: tous les fichiers ({len(to_augment)})")
    
    # Ignorer les déjà augmentés
    if 'augmentation' in df.columns:
        to_augment = to_augment[to_augment['augmentation'].isna()]
        print(f"       {Colors.DIM}→ {len(to_augment)} originaux à traiter{Colors.END}")
    
    if len(to_augment) == 0:
        print_warning("Rien à augmenter")
        return
    
    # Augmentations à appliquer
    augmentations = [
        ("noise", lambda a, sr: add_noise(a)),
        ("slow", lambda a, sr: stretch(a, 0.9)),
        ("fast", lambda a, sr: stretch(a, 1.1)),
        ("pitch_up", lambda a, sr: shift_pitch(a, sr, 2)),
        ("pitch_down", lambda a, sr: shift_pitch(a, sr, -2)),
    ]
    
    total_to_create = len(to_augment) * len(augmentations)
    print()
    print(f"  {Colors.BOLD}Transformations:{Colors.END}")
    for name, _ in augmentations:
        print(f"    • {name}")
    print()
    print_info(f"{total_to_create} fichiers seront créés")
    
    confirm = input(f"\n  Continuer? {Colors.DIM}(o/n){Colors.END} ").strip().lower()
    if confirm not in ['o', 'y', 'oui', 'yes']:
        print_warning("Annulé par l'utilisateur")
        return
    
    next_num = get_next_num()
    new_rows = []
    errors = 0
    
    # Ajouter colonnes si nécessaire
    if 'source' not in df.columns:
        df['source'] = df['nfile']
    if 'augmentation' not in df.columns:
        df['augmentation'] = pd.NA
    
    print()
    total = len(to_augment)
    for idx, (_, row) in enumerate(to_augment.iterrows(), 1):
        path = os.path.join(SLICES_DIR, row['nfile'])
        if not os.path.exists(path):
            continue
        
        # Barre de progression
        progress = int(idx / total * 30)
        bar = f"[{'█' * progress}{'░' * (30 - progress)}]"
        print(f"\r  {bar} {idx}/{total}", end="", flush=True)
        
        try:
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)
            audio = librosa.util.normalize(audio)
        except Exception as e:
            print(f"\n{Colors.YELLOW}  [!] Erreur lecture: {row['nfile']}{Colors.END}")
            errors += 1
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
                errors += 1
    
    print()  # Nouvelle ligne après la barre de progression
    
    # Sauvegarder
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_new], ignore_index=True)
        df_combined.to_csv(ANNOTATION_CSV, index=False)
        
        print()
        print_header("Résultat")
        print_success(f"{len(new_rows)} fichiers créés")
        print_stat("Total en base", len(df_combined))
        if errors > 0:
            print_warning(f"{errors} erreurs rencontrées")


def main():
    if len(sys.argv) < 2:
        print_header("Augmentation de données")
        print()
        print(f"  {Colors.BOLD}1{Colors.END}  Voir le statut")
        print(f"  {Colors.BOLD}2{Colors.END}  Augmenter tout")
        print(f"  {Colors.BOLD}3{Colors.END}  Augmenter un label")
        print(f"  {Colors.DIM}0  Quitter{Colors.END}")
        
        choice = input(f"\n  Choix: ").strip()
        
        if choice == "1":
            show_status()
        elif choice == "2":
            augment()
        elif choice == "3":
            label = input(f"  Label {Colors.DIM}(1 ou 2){Colors.END}: ").strip()
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
            print_error("Commande inconnue")
            print(f"  {Colors.DIM}Usage: python data_augmentation.py [status|augment [label]]{Colors.END}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Annulé par l'utilisateur")
    except ValueError as e:
        print_error(f"Valeur invalide: {e}")
    except Exception as e:
        print_error(f"Erreur inattendue: {e}")
