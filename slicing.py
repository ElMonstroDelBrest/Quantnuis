#!/usr/bin/env python3
"""
Découpage d'un fichier audio en slices

Découpe un fichier audio long selon un fichier d'annotations CSV.

Format du CSV d'annotations attendu :
    Start,End,Label,Reliability
    00:09:34,00:10:12,1,3
    00:11:30,00:11:43,2,3
    ...

Usage:
    python slicing.py audio.wav annotations.csv
"""

import os
import sys
import pandas as pd
from pydub import AudioSegment
from datetime import datetime

# Configuration
SLICES_DIR = "data/slices"
ANNOTATION_CSV = "data/annotation.csv"


def time_to_seconds(time_str: str) -> int:
    """Convertit HH:MM:SS en secondes"""
    t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def get_next_num() -> int:
    """Trouve le prochain numéro de slice"""
    if not os.path.exists(SLICES_DIR):
        return 1
    files = [f for f in os.listdir(SLICES_DIR) if f.startswith('slice_') and f.endswith('.wav')]
    if not files:
        return 1
    nums = [int(f.replace('slice_', '').replace('.wav', '')) for f in files]
    return max(nums) + 1


def slice_audio(audio_path: str, annotations_path: str):
    """Découpe un fichier audio selon les annotations"""
    
    os.makedirs(SLICES_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Charger les annotations existantes
    existing = set()
    if os.path.exists(ANNOTATION_CSV):
        df_existing = pd.read_csv(ANNOTATION_CSV)
        existing = set(df_existing['nfile'].values)
        print(f"📄 {len(df_existing)} annotations existantes")
    else:
        df_existing = None
    
    # Lire les nouvelles annotations
    df_new = pd.read_csv(annotations_path)
    print(f"📥 {len(df_new)} segments à découper")
    
    # Charger l'audio
    print(f"🎵 Chargement de {audio_path}...")
    audio = AudioSegment.from_file(audio_path)
    
    next_num = get_next_num()
    new_rows = []
    
    for _, row in df_new.iterrows():
        start_s = time_to_seconds(row['Start'])
        end_s = time_to_seconds(row['End'])
        
        nfile = f"slice_{next_num:03d}.wav"
        output_path = os.path.join(SLICES_DIR, nfile)
        
        if os.path.exists(output_path):
            print(f"  ⚠ {nfile} existe déjà")
            continue
        
        try:
            segment = audio[start_s * 1000:end_s * 1000]
            segment.export(output_path, format="wav")
            
            new_rows.append({
                'nfile': nfile,
                'length': end_s - start_s,
                'label': row['Label'],
                'reliability': row['Reliability']
            })
            
            print(f"  ✓ {nfile} ({row['Start']} → {row['End']}, label={row['Label']})")
            next_num += 1
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    # Fusionner et sauvegarder
    if new_rows:
        df_add = pd.DataFrame(new_rows)
        if df_existing is not None:
            df_final = pd.concat([df_existing, df_add], ignore_index=True)
        else:
            df_final = df_add
        
        df_final = df_final.drop_duplicates(subset=['nfile']).sort_values('nfile')
        df_final.to_csv(ANNOTATION_CSV, index=False)
        
        print(f"\n✓ {len(new_rows)} slices créés")
        print(f"📊 Total: {len(df_final)} annotations")
    else:
        print("\n⚠ Aucun slice créé")


def main():
    if len(sys.argv) < 3:
        print("Usage: python slicing.py <audio.wav> <annotations.csv>")
        print("\nFormat du CSV:")
        print("  Start,End,Label,Reliability")
        print("  00:09:34,00:10:12,1,3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    annotations_path = sys.argv[2]
    
    if not os.path.exists(audio_path):
        print(f"❌ Fichier audio non trouvé: {audio_path}")
        sys.exit(1)
    
    if not os.path.exists(annotations_path):
        print(f"❌ Fichier annotations non trouvé: {annotations_path}")
        sys.exit(1)
    
    slice_audio(audio_path, annotations_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Annulé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
