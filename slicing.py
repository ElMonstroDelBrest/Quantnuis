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

def verify_consistency(output_dir=SLICES_DIR, output_csv=ANNOTATION_CSV):
    """
    Vérifie la cohérence entre les fichiers audio et les annotations
    Retourne True si tout est cohérent, False sinon
    """
    if not os.path.exists(output_csv):
        return True  # Pas de CSV = pas d'incohérence
    
    # Lire les annotations
    try:
        df = pd.read_csv(output_csv)
        annotated_files = set(df['nfile'].values)
    except Exception as e:
        print(f"⚠ Erreur lors de la lecture de {output_csv}: {e}")
        return False
    
    # Lister les fichiers audio
    if not os.path.exists(output_dir):
        return True
    
    actual_files = set([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    
    # Vérifier les incohérences
    missing_files = annotated_files - actual_files
    unannotated_files = actual_files - annotated_files
    
    if missing_files or unannotated_files:
        if missing_files:
            print(f"  ⚠ {len(missing_files)} fichier(s) annoté(s) mais absent(s) du disque")
        if unannotated_files:
            print(f"  ⚠ {len(unannotated_files)} fichier(s) sur disque mais non annoté(s)")
        return False
    
    return True

def clean_annotations(output_dir=SLICES_DIR, output_csv=ANNOTATION_CSV):
    """
    Nettoie les annotations pour ne garder que celles correspondant aux fichiers existants
    """
    if not os.path.exists(output_csv):
        return
    
    try:
        df = pd.read_csv(output_csv)
    except Exception:
        return
    
    # Lister les fichiers audio existants
    if not os.path.exists(output_dir):
        return
    
    actual_files = set([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    
    # Filtrer pour ne garder que les annotations correspondant aux fichiers existants
    df_cleaned = df[df['nfile'].isin(actual_files)].copy()
    
    # Supprimer les doublons (garder la première occurrence)
    df_cleaned = df_cleaned.drop_duplicates(subset=['nfile'], keep='first')
    
    # Trier par nom de fichier
    df_cleaned = df_cleaned.sort_values('nfile').reset_index(drop=True)
    
    # Sauvegarder si des changements ont été faits
    if len(df_cleaned) != len(df) or len(df_cleaned) != len(df_cleaned.drop_duplicates(subset=['nfile'])):
        df_cleaned.to_csv(output_csv, index=False)
        removed = len(df) - len(df_cleaned)
        if removed > 0:
            print(f"  ✓ Nettoyage: {removed} annotation(s) supprimée(s)")

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
        try:
            audio_slice = audio[start_ms:end_ms]
            
            # Exporter le slice
            audio_slice.export(output_path, format="wav")
            
            # Vérifier que le fichier a bien été créé
            if not os.path.exists(output_path):
                print(f"❌ Erreur: {nfile} n'a pas été créé correctement")
                skipped_count += 1
                continue
            
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
        except Exception as e:
            print(f"❌ Erreur lors de la création de {nfile}: {e}")
            skipped_count += 1
            # Supprimer le fichier s'il a été partiellement créé
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            continue
    
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
        
        # Nettoyer les doublons avant de sauvegarder
        df_output = df_output.drop_duplicates(subset=['nfile'], keep='first')
        df_output = df_output.sort_values('nfile').reset_index(drop=True)
        
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
    
    # Nettoyer les annotations pour s'assurer de la cohérence
    print("\n" + "="*60)
    print("VÉRIFICATION ET NETTOYAGE DE LA BASE DE DONNÉES")
    print("="*60)
    
    # Vérifier que tous les nouveaux fichiers sont bien dans le CSV
    if new_annotation_data:
        created_files = set([item['nfile'] for item in new_annotation_data])
        try:
            df_final = pd.read_csv(output_csv)
            csv_files = set(df_final['nfile'].values)
            missing_in_csv = created_files - csv_files
            if missing_in_csv:
                print(f"⚠ {len(missing_in_csv)} fichier(s) créé(s) mais non trouvé(s) dans le CSV")
                print("  Réintégration en cours...")
                # Réintégrer les fichiers manquants
                missing_data = [item for item in new_annotation_data if item['nfile'] in missing_in_csv]
                if missing_data:
                    df_missing = pd.DataFrame(missing_data)
                    df_final = pd.concat([df_final, df_missing], ignore_index=True)
                    df_final = df_final.drop_duplicates(subset=['nfile'], keep='first')
                    df_final = df_final.sort_values('nfile').reset_index(drop=True)
                    df_final.to_csv(output_csv, index=False)
                    print(f"  ✓ {len(missing_data)} annotation(s) réintégrée(s)")
        except Exception as e:
            print(f"  ⚠ Erreur lors de la vérification: {e}")
    
    clean_annotations(output_dir, output_csv)
    
    # Vérifier la cohérence finale
    if verify_consistency(output_dir, output_csv):
        print("✓ Base de données cohérente: tous les fichiers audio ont des annotations")
    else:
        print("⚠ Incohérences détectées, nettoyage en cours...")
        clean_annotations(output_dir, output_csv)
        if verify_consistency(output_dir, output_csv):
            print("✓ Base de données corrigée et cohérente")
        else:
            print("⚠ Des incohérences persistent, vérifiez manuellement")
    
    # Afficher le résumé final
    try:
        df_final = pd.read_csv(output_csv)
        actual_count = len([f for f in os.listdir(output_dir) if f.endswith('.wav')]) if os.path.exists(output_dir) else 0
        print(f"\n📊 Résumé final:")
        print(f"   Fichiers audio: {actual_count}")
        print(f"   Annotations: {len(df_final)}")
        print(f"   Cohérence: {'✓' if actual_count == len(df_final) else '⚠'}")
    except:
        pass
    
    print(f"\n✓ Tous les slices sont dans le répertoire '{output_dir}'")
    print(f"✓ Base de données mise à jour: {output_csv}")

if __name__ == "__main__":
    import sys
    
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) < 3:
        print("ERREUR: Arguments manquants.")
        print("Usage: python slicing.py <chemin_vers_fichier_audio> <chemin_vers_fichier_annotations>")
        print("\nExemple:")
        print("  python slicing.py data/audio.wav data/raw/annotations_raw.csv")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    annotations_file = sys.argv[2]
    
    # Vérifier que le fichier audio existe
    if not os.path.exists(input_audio):
        print(f"ERREUR: Le fichier audio '{input_audio}' n'existe pas.")
        sys.exit(1)
    
    # Vérifier que le fichier d'annotations existe
    if not os.path.exists(annotations_file):
        print(f"ERREUR: Le fichier d'annotations '{annotations_file}' n'existe pas.")
        sys.exit(1)
    
    print(f"Utilisation du fichier audio: {input_audio}")
    print(f"Utilisation du fichier d'annotations: {annotations_file}")
    slice_audio(input_audio, annotations_file)
