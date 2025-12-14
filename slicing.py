import pandas as pd
import os
from pydub import AudioSegment
from datetime import datetime

# Configuration des chemins
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
SLICES_DIR = os.path.join(DATA_DIR, "slices")
ANNOTATION_CSV = os.path.join(DATA_DIR, "annotation.csv")

# Créer les dossiers s'ils n'existent pas
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SLICES_DIR, exist_ok=True)

def time_to_seconds(time_str):
    """Convertit un timestamp HH:MM:SS en secondes"""
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def get_next_slice_number(slices_dir):
    """
    Trouve le prochain numéro de slice disponible en vérifiant les fichiers existants
    """
    if not os.path.exists(slices_dir):
        return 1
    
    existing_files = [f for f in os.listdir(slices_dir) if f.startswith('slice_') and f.endswith('.wav')]
    if not existing_files:
        return 1
    
    # Extraire les numéros des fichiers existants
    numbers = []
    for f in existing_files:
        try:
            # Format: slice_XXX.wav
            num_str = f.replace('slice_', '').replace('.wav', '')
            numbers.append(int(num_str))
        except ValueError:
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def slice_audio(input_audio_path, annotations_csv, output_dir=SLICES_DIR, output_csv=ANNOTATION_CSV):
    """
    Découpe un fichier audio selon les annotations et ajoute au fichier annotation.csv existant
    Évite les doublons par nom de fichier
    
    Args:
        input_audio_path: Chemin vers le fichier audio source
        annotations_csv: Chemin vers le fichier annotations_raw.csv
        output_dir: Répertoire où sauvegarder les slices audio
        output_csv: Nom du fichier CSV de sortie
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le CSV d'annotations existant s'il existe
    existing_df = None
    existing_files = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_files = set(existing_df['nfile'].values)
            print(f"✓ Fichier d'annotations existant trouvé: {len(existing_df)} entrées")
        except Exception as e:
            print(f"⚠ Impossible de lire le fichier d'annotations existant: {e}")
            existing_df = None
            existing_files = set()
    
    # Lire les nouvelles annotations
    df_annotations = pd.read_csv(annotations_csv)
    
    # Charger le fichier audio
    print(f"Chargement du fichier audio: {input_audio_path}")
    audio = AudioSegment.from_file(input_audio_path)
    
    # Trouver le prochain numéro de slice disponible
    next_slice_num = get_next_slice_number(output_dir)
    print(f"✓ Prochain numéro de slice: {next_slice_num}")
    
    # Liste pour stocker les nouvelles données
    new_annotation_data = []
    skipped_count = 0
    added_count = 0
    
    # Parcourir chaque annotation
    for idx, row in df_annotations.iterrows():
        start_time = row['Start']
        end_time = row['End']
        label = row['Label']
        reliability = row['Reliability']
        
        # Convertir les timestamps en secondes puis en millisecondes
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        start_ms = start_seconds * 1000
        end_ms = end_seconds * 1000
        
        # Calculer la longueur en secondes
        length = end_seconds - start_seconds
        
        # Nom du fichier de sortie (format: slice_XXX.wav)
        nfile = f"slice_{next_slice_num:03d}.wav"
        
        # Vérifier si le fichier existe déjà (doublon)
        if nfile in existing_files:
            print(f"⚠ Doublon détecté: {nfile} existe déjà, ignoré")
            skipped_count += 1
            continue
        
        output_path = os.path.join(output_dir, nfile)
        
        # Vérifier si le fichier audio existe déjà sur le disque
        if os.path.exists(output_path):
            print(f"⚠ Fichier existant: {nfile}, ignoré")
            skipped_count += 1
            continue
        
        # Extraire le slice audio
        audio_slice = audio[start_ms:end_ms]
        
        # Exporter le slice
        audio_slice.export(output_path, format="wav")
        print(f"✓ Slice {next_slice_num} sauvegardé: {nfile} ({start_time} -> {end_time}, label={label})")
        
        # Ajouter les données pour le CSV
        new_annotation_data.append({
            'nfile': nfile,
            'length': length,
            'label': label,
            'reliability': reliability
        })
        
        added_count += 1
        next_slice_num += 1
    
    # Fusionner avec les données existantes
    if new_annotation_data:
        df_new = pd.DataFrame(new_annotation_data)
        
        if existing_df is not None:
            # Concaténer avec les données existantes
            df_output = pd.concat([existing_df, df_new], ignore_index=True)
            print(f"\n✓ {added_count} nouveaux slices ajoutés")
        else:
            # Première création du fichier
            df_output = df_new
            print(f"\n✓ {added_count} slices créés")
        
        # Sauvegarder le CSV mis à jour
        df_output.to_csv(output_csv, index=False)
        print(f"✓ Fichier {output_csv} mis à jour: {len(df_output)} annotations au total")
    else:
        if existing_df is not None:
            df_output = existing_df
            print(f"\n⚠ Aucun nouveau slice ajouté (tous étaient des doublons)")
        else:
            print(f"\n⚠ Aucun slice créé")
    
    if skipped_count > 0:
        print(f"⚠ {skipped_count} slice(s) ignoré(s) (doublons)")
    
    print(f"✓ Tous les slices sont dans le répertoire '{output_dir}'")

if __name__ == "__main__":
    import sys
    
    # Configuration - chercher un fichier annotations_raw.csv dans data/raw
    annotations_file = None
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('_annotations_raw.csv') or f == 'annotations_raw.csv'] if os.path.exists(RAW_DIR) else []
    if raw_files:
        annotations_file = os.path.join(RAW_DIR, sorted(raw_files)[-1])  # Prendre le plus récent
    else:
        # Chercher dans le répertoire courant si pas trouvé
        for f in os.listdir('.'):
            if f.endswith('_annotations_raw.csv') or f == 'annotations_raw.csv':
                annotations_file = f
                break
    
    if not annotations_file or not os.path.exists(annotations_file):
        print("ERREUR: Aucun fichier annotations_raw.csv trouvé.")
        print(f"Cherchez dans '{RAW_DIR}' ou le répertoire courant.")
        print("Vous pouvez créer un fichier avec annotation.py")
        sys.exit(1)
    
    # Chercher automatiquement un fichier audio dans le répertoire courant
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    input_audio = None
    
    # Si un argument est fourni en ligne de commande, l'utiliser
    if len(sys.argv) > 1:
        input_audio = sys.argv[1]
    else:
        # Sinon, chercher automatiquement un fichier audio
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                input_audio = file
                break
    
    # Vérifier que le fichier audio existe
    if not input_audio or not os.path.exists(input_audio):
        print("ERREUR: Aucun fichier audio trouvé.")
        print("Usage: python slicing.py [chemin_vers_fichier_audio]")
        print("Ou placez un fichier audio (.wav, .mp3, .flac, etc.) dans le répertoire courant.")
        sys.exit(1)
    
    print(f"Utilisation du fichier audio: {input_audio}")
    print(f"Utilisation du fichier d'annotations: {annotations_file}")
    slice_audio(input_audio, annotations_file)
