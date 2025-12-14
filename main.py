#!/usr/bin/env python3
"""
Script principal pour le système de classification audio
Permet de lancer facilement les différentes étapes du workflow
"""

import os
import sys
import subprocess

def print_menu():
    """Affiche le menu principal"""
    print("\n" + "="*60)
    print("SYSTÈME DE CLASSIFICATION AUDIO")
    print("="*60)
    print("\nChoisissez une action :")
    print("  1. Créer le fichier d'annotations (annotation.py)")
    print("  2. Découper un fichier audio en segments (slicing.py)")
    print("  3. Entraîner le modèle (model_improved.py)")
    print("  4. Prédire la classe d'un fichier audio (predict_improved.py)")
    print("  5. Analyser un fichier audio complet (predict_full_audio.py)")
    print("  6. Fusionner tous les slices existants dans data/slices/")
    print("  0. Quitter")
    print("="*60)

def run_annotation():
    """Lance le script d'annotation"""
    print("\n📝 Création du fichier d'annotations...")
    python_exe = sys.executable
    subprocess.run([python_exe, "annotation.py"], check=False)

def run_slicing():
    """Lance le script de découpage"""
    print("\n✂️  Découpage du fichier audio...")
    if len(sys.argv) > 2:
        audio_file = sys.argv[2]
    else:
        audio_file = input("Entrez le chemin vers le fichier audio (ou appuyez sur Entrée pour chercher automatiquement): ").strip()
        if not audio_file:
            audio_file = None
    
    python_exe = sys.executable
    if audio_file:
        subprocess.run([python_exe, "slicing.py", audio_file], check=False)
    else:
        subprocess.run([python_exe, "slicing.py"], check=False)

def run_training():
    """Lance l'entraînement du modèle"""
    print("\n🎓 Entraînement du modèle...")
    print("Cela peut prendre du temps selon la taille de votre dataset.")
    python_exe = sys.executable
    subprocess.run([python_exe, "model_improved.py"], check=False)

def run_predict():
    """Lance la prédiction sur un fichier"""
    print("\n🔮 Prédiction sur un fichier audio...")
    if len(sys.argv) > 2:
        audio_file = sys.argv[2]
    else:
        audio_file = input("Entrez le chemin vers le fichier audio (ou appuyez sur Entrée pour utiliser le premier fichier de data/slices/): ").strip()
        if not audio_file:
            audio_file = None
    
    python_exe = sys.executable
    if audio_file:
        subprocess.run([python_exe, "predict_improved.py", audio_file], check=False)
    else:
        subprocess.run([python_exe, "predict_improved.py"], check=False)

def run_full_analysis():
    """Lance l'analyse d'un fichier audio complet"""
    print("\n📊 Analyse d'un fichier audio complet...")
    if len(sys.argv) > 2:
        audio_file = sys.argv[2]
    else:
        audio_file = input("Entrez le chemin vers le fichier audio (ou appuyez sur Entrée pour chercher automatiquement): ").strip()
        if not audio_file:
            audio_file = None
    
    python_exe = sys.executable
    if audio_file:
        subprocess.run([python_exe, "predict_full_audio.py", audio_file], check=False)
    else:
        subprocess.run([python_exe, "predict_full_audio.py"], check=False)

def run_merge_slices():
    """Fusionne tous les slices existants dans data/slices/"""
    print("\n🔄 Fusion de tous les slices...")
    python_exe = sys.executable
    
    # Exécuter directement sans capture pour voir la sortie en temps réel
    try:
        subprocess.run([python_exe, "merge_slices.py"], check=False)
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Fonction principale"""
    # Vérifier que pandas est disponible (indicateur que le venv est activé)
    try:
        import pandas
    except ImportError:
        print("\n⚠ ATTENTION: pandas n'est pas disponible!")
        print(f"   Python utilisé: {sys.executable}")
        print("   Assurez-vous d'avoir activé votre venv:")
        print("   source venv/bin/activate  # ou votre chemin vers le venv")
        print("   pip install pandas")
        sys.exit(1)
    
    # Vérifier que les dossiers existent
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/slices", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Si un argument est fourni, l'utiliser directement
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print_menu()
        choice = input("\nVotre choix: ").strip()
    
    if choice == "1":
        run_annotation()
    elif choice == "2":
        run_slicing()
    elif choice == "3":
        run_training()
    elif choice == "4":
        run_predict()
    elif choice == "5":
        run_full_analysis()
    elif choice == "6":
        run_merge_slices()
    elif choice == "0":
        print("\nAu revoir !")
        sys.exit(0)
    elif choice == "help":
        print("Usage: python main.py [choix]")
        print("1 : Créer le fichier d'annotations (annotation.py)")
        print("2 : Découper un fichier audio en segments (slicing.py)")
        print("3 : Entraîner le modèle (model_improved.py)")
        print("4 : Prédire la classe d'un fichier audio (predict_improved.py)")
        print("5 : Analyser un fichier audio complet (predict_full_audio.py)")
        print("6 : Fusionner tous les slices existants dans data/slices/")
        print("0 : Quitter")
    else:
        print("\n❌ Choix invalide. Veuillez choisir un nombre entre 0 et 6.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
