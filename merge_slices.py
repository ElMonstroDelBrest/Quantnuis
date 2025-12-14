#!/usr/bin/env python3
"""
Script pour fusionner tous les slices de différents dossiers dans data/slices/
et fusionner tous les fichiers d'annotations en un seul data/annotation.csv
"""

import os
import pandas as pd
import shutil

DATA_DIR = "data"
SLICES_DIR = os.path.join(DATA_DIR, "slices")
ANNOTATION_CSV = os.path.join(DATA_DIR, "annotation.csv")

def get_next_slice_number(slices_dir):
    """Trouve le prochain numéro de slice disponible"""
    if not os.path.exists(slices_dir):
        return 1
    
    existing_files = [f for f in os.listdir(slices_dir) if f.startswith('slice_') and f.endswith('.wav')]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            num_str = f.replace('slice_', '').replace('.wav', '')
            numbers.append(int(num_str))
        except ValueError:
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def merge_all_slices():
    """Fusionne tous les slices de différents dossiers dans data/slices/"""
    print("="*60)
    print("FUSION DE TOUS LES SLICES")
    print("="*60)
    
    # Créer le dossier de destination
    os.makedirs(SLICES_DIR, exist_ok=True)
    
    # Chercher tous les dossiers slices (sauf data/slices)
    slices_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('slices'):
            # Exclure le dossier "slices" à la racine seulement si data/slices existe
            # (car on veut fusionner slices1, slices2, etc. dans data/slices)
            if item == "slices":
                # Si data/slices existe, on ignore le dossier "slices" à la racine
                if os.path.exists(SLICES_DIR):
                    continue
            slices_dirs.append(item)
    
    if not slices_dirs:
        print("Aucun dossier slices trouvé à fusionner (sauf data/slices/)")
        return
    
    print(f"\nDossiers trouvés: {slices_dirs}")
    
    # Charger le CSV d'annotations existant
    all_annotations = []
    existing_files_in_csv = set()
    if os.path.exists(ANNOTATION_CSV):
        try:
            existing_df = pd.read_csv(ANNOTATION_CSV)
            all_annotations = existing_df.to_dict('records')
            existing_files_in_csv = set(existing_df['nfile'].values)
            print(f"✓ {len(all_annotations)} annotations existantes chargées depuis {ANNOTATION_CSV}")
        except Exception as e:
            print(f"⚠ Erreur lors du chargement de {ANNOTATION_CSV}: {e}")
    
    # Obtenir la liste des fichiers existants dans data/slices/
    existing_files = set()
    if os.path.exists(SLICES_DIR):
        existing_files = set([f for f in os.listdir(SLICES_DIR) if f.endswith('.wav')])
        print(f"✓ {len(existing_files)} fichiers audio existants dans {SLICES_DIR}")
    
    total_copied = 0
    total_skipped = 0
    
    # Parcourir chaque dossier slices
    for slices_dir in slices_dirs:
        print(f"\n📁 Traitement de {slices_dir}/")
        
        if not os.path.exists(slices_dir):
            print(f"  ⚠ Dossier {slices_dir} n'existe pas")
            continue
        
        # Chercher les fichiers CSV d'annotations associés
        # Chercher annotation.csv, annotation1.csv, annotation2.csv, etc.
        annotation_files = []
        base_name = slices_dir.replace('slices', 'annotation')
        possible_names = [
            'annotation.csv',
            f'{base_name}.csv',
            base_name + '1.csv' if base_name == 'annotation' else base_name + '.csv'
        ]
        
        # Chercher aussi tous les fichiers annotation*.csv
        for csv_file in os.listdir('.'):
            if csv_file.startswith('annotation') and csv_file.endswith('.csv'):
                annotation_files.append(csv_file)
        
        # Lire les annotations de ce dossier si elles existent
        dir_annotations = {}
        for csv_file in annotation_files:
            try:
                if not os.path.exists(csv_file):
                    continue
                df = pd.read_csv(csv_file)
                if not df.empty and 'nfile' in df.columns:
                    for _, row in df.iterrows():
                        nfile = row['nfile']
                        if pd.notna(nfile):  # Ignorer les valeurs NaN
                            dir_annotations[nfile] = row.to_dict()
            except Exception as e:
                print(f"  ⚠ Erreur lors de la lecture de {csv_file}: {e}")
        
        if dir_annotations:
            print(f"  ✓ {len(dir_annotations)} annotations trouvées")
        
        # Copier les fichiers audio
        files_in_dir = [f for f in os.listdir(slices_dir) if f.endswith('.wav')]
        print(f"  📊 {len(files_in_dir)} fichier(s) .wav trouvé(s)")
        
        for file in files_in_dir:
            source_path = os.path.join(slices_dir, file)
            
            # Vérifier si le fichier existe déjà (par nom)
            if file in existing_files:
                print(f"  ⚠ {file} existe déjà dans {SLICES_DIR}, ignoré")
                total_skipped += 1
                continue
            
            # Copier vers data/slices/
            try:
                dest_path = os.path.join(SLICES_DIR, file)
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ {file} copié")
                total_copied += 1
                existing_files.add(file)
            except Exception as e:
                print(f"  ❌ Erreur lors de la copie de {file}: {e}")
                continue
            
            # Ajouter l'annotation si elle existe
            if file in dir_annotations:
                # Vérifier si l'annotation existe déjà (doublon)
                if file not in existing_files_in_csv:
                    all_annotations.append(dir_annotations[file])
                    existing_files_in_csv.add(file)
                else:
                    print(f"    ⚠ Annotation pour {file} existe déjà dans le CSV, ignorée")
    
    # Sauvegarder le CSV d'annotations fusionné
    if all_annotations:
        # Supprimer les doublons par nom de fichier (garder la première occurrence)
        seen_files = set()
        unique_annotations = []
        for ann in all_annotations:
            if ann['nfile'] not in seen_files:
                seen_files.add(ann['nfile'])
                unique_annotations.append(ann)
        
        df_final = pd.DataFrame(unique_annotations)
        # Vérifier que le DataFrame n'est pas vide et a les bonnes colonnes
        if not df_final.empty and 'nfile' in df_final.columns:
            # Trier par nom de fichier pour une meilleure organisation
            df_final = df_final.sort_values('nfile')
        df_final.to_csv(ANNOTATION_CSV, index=False)
        print(f"\n✓ {len(unique_annotations)} annotations uniques sauvegardées dans {ANNOTATION_CSV}")
    else:
        print(f"\n⚠ Aucune annotation à sauvegarder")
    
    print(f"\n{'='*60}")
    print(f"Résumé:")
    print(f"  ✓ {total_copied} fichier(s) copié(s)")
    print(f"  ⚠ {total_skipped} fichier(s) ignoré(s) (doublons)")
    print(f"  📁 Tous les slices sont maintenant dans {SLICES_DIR}")
    print(f"  📄 Toutes les annotations sont dans {ANNOTATION_CSV}")
    
    # Déplacer les anciens dossiers slices dans un dossier "old"
    if slices_dirs:
        old_dir = "old"
        os.makedirs(old_dir, exist_ok=True)
        moved_count = 0
        
        print(f"\n📦 Déplacement des anciens dossiers dans {old_dir}/...")
        for slices_dir in slices_dirs:
            if os.path.exists(slices_dir):
                try:
                    dest_path = os.path.join(old_dir, slices_dir)
                    # Si le dossier existe déjà dans old, le renommer avec un timestamp
                    if os.path.exists(dest_path):
                        import time
                        timestamp = int(time.time())
                        dest_path = os.path.join(old_dir, f"{slices_dir}_{timestamp}")
                    
                    shutil.move(slices_dir, dest_path)
                    print(f"  ✓ {slices_dir} déplacé vers {dest_path}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ⚠ Erreur lors du déplacement de {slices_dir}: {e}")
        
        if moved_count > 0:
            print(f"  ✓ {moved_count} dossier(s) déplacé(s) dans {old_dir}/")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        merge_all_slices()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur lors de la fusion: {e}")
        import traceback
        traceback.print_exc()
