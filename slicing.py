import pandas as pd
import os
from pydub import AudioSegment
from datetime import datetime

def time_to_seconds(time_str):
    """Convertit un timestamp HH:MM:SS en secondes"""
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def slice_audio(input_audio_path, annotations_csv, output_dir="slices", output_csv="annotation.csv"):
    """
    Découpe un fichier audio selon les annotations et crée un fichier annotation.csv
    
    Args:
        input_audio_path: Chemin vers le fichier audio source
        annotations_csv: Chemin vers le fichier annotations_raw.csv
        output_dir: Répertoire où sauvegarder les slices audio
        output_csv: Nom du fichier CSV de sortie
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Lire les annotations
    df_annotations = pd.read_csv(annotations_csv)
    
    # Charger le fichier audio
    print(f"Chargement du fichier audio: {input_audio_path}")
    audio = AudioSegment.from_file(input_audio_path)
    
    # Liste pour stocker les données du nouveau CSV
    annotation_data = []
    
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
        
        # Extraire le slice audio
        audio_slice = audio[start_ms:end_ms]
        
        # Nom du fichier de sortie (format: slice_001.wav, slice_002.wav, etc.)
        nfile = f"slice_{idx+1:03d}.wav"
        output_path = os.path.join(output_dir, nfile)
        
        # Exporter le slice
        audio_slice.export(output_path, format="wav")
        print(f"Slice {idx+1} sauvegardé: {nfile} ({start_time} -> {end_time}, label={label})")
        
        # Ajouter les données pour le CSV
        annotation_data.append({
            'nfile': nfile,
            'length': length,
            'label': label,
            'reliability': reliability
        })
    
    # Créer le DataFrame et sauvegarder dans annotation.csv
    df_output = pd.DataFrame(annotation_data)
    df_output.to_csv(output_csv, index=False)
    print(f"\nFichier {output_csv} créé avec {len(annotation_data)} annotations")
    print(f"Tous les slices ont été sauvegardés dans le répertoire '{output_dir}'")

if __name__ == "__main__":
    import sys
    
    # Configuration
    annotations_file = "20250928_122537_annotations_raw.csv"
    output_directory = "slices2"
    output_annotation_csv = "annotation2.csv"
    
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
    slice_audio(input_audio, annotations_file, output_directory, output_annotation_csv)
