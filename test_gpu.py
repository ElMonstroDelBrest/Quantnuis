#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration GPU TensorFlow
"""

import tensorflow as tf
import sys

print("="*60)
print("TEST CONFIGURATION GPU TENSORFLOW")
print("="*60)

# Version TensorFlow
print(f"\nTensorFlow version: {tf.__version__}")

# Vérifier CUDA
print(f"CUDA disponible: {tf.test.is_built_with_cuda()}")

# Lister les GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU(s) détecté(s): {len(gpus)}")

if gpus:
    print("\n✓ GPU(s) trouvé(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Détails: {details}")
        except:
            pass
    
    # Configuration de la mémoire GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n✓ Configuration mémoire GPU activée (croissance dynamique)")
    except RuntimeError as e:
        print(f"\n⚠ Erreur configuration mémoire: {e}")
    
    # Test simple avec GPU
    print("\n🧪 Test d'opération sur GPU...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"✓ Test réussi! Résultat: {c.numpy()}")
            print(f"  Device utilisé: {c.device}")
    except Exception as e:
        print(f"❌ Erreur lors du test GPU: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ VOTRE GPU EST CONFIGURÉ ET FONCTIONNE!")
    print("="*60)
    print("\nVous pouvez maintenant entraîner votre modèle avec:")
    print("  python main.py 3")
    print("\nPour surveiller l'utilisation GPU pendant l'entraînement:")
    print("  watch -n 1 nvidia-smi")
    
else:
    print("\n⚠ AUCUN GPU DÉTECTÉ")
    print("\nVérifications à faire:")
    print("  1. Vérifier les drivers NVIDIA: nvidia-smi")
    print("  2. Installer TensorFlow GPU: pip install tensorflow[and-cuda]")
    print("  3. Vérifier CUDA: nvcc --version")
    print("\nConsultez SETUP_GPU.md pour plus de détails")
    sys.exit(1)
