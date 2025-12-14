# Configuration GPU NVIDIA pour TensorFlow

Ce guide explique comment configurer TensorFlow pour utiliser votre GPU NVIDIA.

> **📖 Pour Ubuntu/Debian :** Consultez `INSTALL_GPU_UBUNTU.md` pour un guide simplifié spécifique à Ubuntu.

## 📋 Prérequis

1. **GPU NVIDIA compatible** avec CUDA
2. **Drivers NVIDIA** installés
3. **TensorFlow avec support GPU** (CUDA/cuDNN installés automatiquement)

## 🔧 Installation Rapide (Recommandée)

### Méthode Simple : Installation Automatique

TensorFlow 2.13+ peut installer automatiquement CUDA et cuDNN :

```bash
# Activer votre venv
source venv/bin/activate

# Désinstaller TensorFlow CPU si installé
pip uninstall tensorflow tensorflow-cpu

# Installer TensorFlow avec CUDA/cuDNN automatique
pip install tensorflow[and-cuda]
```

**C'est tout !** Cette commande installe automatiquement :
- CUDA Toolkit
- cuDNN
- TensorFlow avec support GPU

Vous n'avez **PAS besoin** d'installer CUDA manuellement.

### Étape 1 : Vérifier votre GPU

```bash
nvidia-smi
```

Cela devrait afficher des informations sur votre GPU. Si vous voyez une erreur, installez d'abord les drivers NVIDIA (voir `INSTALL_GPU_UBUNTU.md`).

### Étape 2 : Installer TensorFlow avec support GPU

Dans votre environnement virtuel :

```bash
# Activer votre venv
source venv/bin/activate  # ou votre chemin

# Désinstaller TensorFlow CPU si installé
pip uninstall tensorflow tensorflow-cpu

# Installer TensorFlow avec support GPU (installe CUDA automatiquement)
pip install tensorflow[and-cuda]
```

**Alternative (version spécifique) :**

```bash
# Pour TensorFlow 2.13+
pip install tensorflow[and-cuda]

# Ou pour une version spécifique avec CUDA 11.8
pip install tensorflow==2.13.0
```

### Étape 3 : Vérifier l'installation

Créez un script de test :

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
print("CUDA disponible:", tf.test.is_built_with_cuda())

# Afficher les détails du GPU
if tf.config.list_physical_devices('GPU'):
    print("\nGPU détecté:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  - {gpu}")
        print(f"    Nom: {tf.config.experimental.get_device_details(gpu)}")
else:
    print("\n⚠ Aucun GPU détecté")
```

Exécutez-le :
```bash
python test_gpu.py
```

## ✅ Vérification Rapide

Dans un terminal Python :

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

Si vous voyez une liste de GPU, c'est bon ! Sinon, consultez la section dépannage.

## 🚀 Utilisation

Une fois configuré, TensorFlow utilisera automatiquement le GPU pour :
- L'entraînement du modèle (`model_improved.py`)
- Les prédictions avec de gros batchs

Vous pouvez forcer l'utilisation du GPU dans votre code :

```python
import tensorflow as tf

# Forcer l'utilisation du GPU
with tf.device('/GPU:0'):
    # Votre code TensorFlow ici
    model.fit(...)
```

## 🔍 Vérifier pendant l'entraînement

Pendant l'entraînement, ouvrez un autre terminal et lancez :

```bash
watch -n 1 nvidia-smi
```

Vous devriez voir l'utilisation du GPU augmenter.

## 🆘 Dépannage

### Problème : "No GPU devices found"

**Solutions :**

1. **Vérifier les drivers NVIDIA :**
   ```bash
   nvidia-smi
   ```
   Si ça ne fonctionne pas, installez les drivers NVIDIA.

2. **Vérifier CUDA :**
   ```bash
   nvcc --version
   ```
   Si ça ne fonctionne pas, installez CUDA Toolkit.

3. **Réinstaller TensorFlow :**
   ```bash
   pip uninstall tensorflow tensorflow-cpu
   pip install tensorflow[and-cuda]
   ```

4. **Vérifier la compatibilité des versions :**
   - TensorFlow 2.13+ : CUDA 11.8 ou 12.x
   - TensorFlow 2.10-2.12 : CUDA 11.2
   
   Consultez : https://www.tensorflow.org/install/source#gpu

### Problème : "Could not load dynamic library"

Cela signifie que TensorFlow ne trouve pas les bibliothèques CUDA/cuDNN.

**Solution :**
- Vérifiez que CUDA et cuDNN sont installés
- Vérifiez que les chemins sont dans `LD_LIBRARY_PATH`
- Réinstallez avec `pip install tensorflow[and-cuda]` qui installe automatiquement les dépendances

### Problème : GPU détecté mais pas utilisé

**Solutions :**

1. **Vérifier la mémoire GPU :**
   ```python
   import tensorflow as tf
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           print(e)
   ```

2. **Forcer l'utilisation du GPU :**
   ```python
   with tf.device('/GPU:0'):
       # Votre code
   ```

## 📊 Performance

Avec GPU, vous devriez voir :
- **10-100x plus rapide** pour l'entraînement
- Utilisation du GPU visible dans `nvidia-smi`
- Messages TensorFlow indiquant l'utilisation du GPU

## 🔗 Ressources

- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [CUDA Installation](https://developer.nvidia.com/cuda-downloads)
- [cuDNN Installation](https://developer.nvidia.com/cudnn)
