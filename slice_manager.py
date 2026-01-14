#!/usr/bin/env python3
"""
Gestionnaire de slices audio

Structure simple :
    data/
    ├── slices/         ← Fichiers audio .wav
    └── annotation.csv  ← Annotations

Usage:
    python slice_manager.py          # Afficher le statut
    python slice_manager.py add      # Ajouter des slices depuis un dossier externe
"""

import os
import sys
import shutil
import pandas as pd

# Configuration
SLICES_DIR = "data/slices"
ANNOTATION_CSV = "data/annotation.csv"


def show_status():
    """Affiche le statut de la base de données"""
    print("=" * 50)
    print("STATUT DE LA BASE DE DONNÉES")
    print("=" * 50)
    
    # Vérifier les slices
    if os.path.exists(SLICES_DIR):
        files = [f for f in os.listdir(SLICES_DIR) if f.endswith('.wav')]
        print(f"\n📁 Slices: {len(files)} fichiers audio")
    else:
        print(f"\n📁 Slices: dossier inexistant")
        return
    
    # Vérifier les annotations
    if os.path.exists(ANNOTATION_CSV):
        df = pd.read_csv(ANNOTATION_CSV)
        print(f"📄 Annotations: {len(df)} entrées")
        
        # Distribution des labels
        if 'label' in df.columns:
            print("\n📊 Distribution:")
            for label in sorted(df['label'].unique()):
                count = (df['label'] == label).sum()
                pct = count / len(df) * 100
                print(f"    Label {label}: {count} ({pct:.1f}%)")
        
        # Vérifier la cohérence
        annotated = set(df['nfile'].values)
        on_disk = set(files)
        
        missing_files = annotated - on_disk
        missing_annotations = on_disk - annotated
        
        if missing_files:
            print(f"\n⚠ {len(missing_files)} fichiers annotés manquants sur disque")
        if missing_annotations:
            print(f"\n⚠ {len(missing_annotations)} fichiers non annotés")
    else:
        print(f"📄 Annotations: fichier inexistant")
    
    print("\n" + "=" * 50)


def add_slices(source_dir: str):
    """Ajoute des slices depuis un dossier externe"""
    if not os.path.exists(source_dir):
        print(f"❌ Dossier non trouvé: {source_dir}")
        return
    
    os.makedirs(SLICES_DIR, exist_ok=True)
    
    # Trouver le prochain numéro
    existing = [f for f in os.listdir(SLICES_DIR) if f.startswith('slice_') and f.endswith('.wav')]
    if existing:
        nums = [int(f.replace('slice_', '').replace('.wav', '')) for f in existing]
        next_num = max(nums) + 1
    else:
        next_num = 1
    
    # Copier les fichiers
    files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    print(f"📥 {len(files)} fichiers à ajouter...")
    
    added = 0
    for f in sorted(files):
        src = os.path.join(source_dir, f)
        new_name = f"slice_{next_num:03d}.wav"
        dst = os.path.join(SLICES_DIR, new_name)
        
        shutil.copy2(src, dst)
        print(f"  ✓ {f} → {new_name}")
        next_num += 1
        added += 1
    
    print(f"\n✓ {added} fichiers ajoutés")
    print(f"⚠ N'oubliez pas de mettre à jour {ANNOTATION_CSV}")


def main():
    if len(sys.argv) < 2:
        show_status()
    elif sys.argv[1] == "add":
        if len(sys.argv) < 3:
            source = input("Dossier source: ").strip()
        else:
            source = sys.argv[2]
        add_slices(source)
    elif sys.argv[1] == "status":
        show_status()
    else:
        print("Usage:")
        print("  python slice_manager.py          # Afficher le statut")
        print("  python slice_manager.py add DIR  # Ajouter des slices")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Annulé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
