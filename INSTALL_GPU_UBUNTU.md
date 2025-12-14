# Installation GPU NVIDIA sur Ubuntu/Debian

Guide simplifié pour installer CUDA et TensorFlow GPU sur Ubuntu/Debian.

> **⚠️ IMPORTANT : Si vous voyez "NVIDIA-SMI has failed" :** 
> 
> **Vous devez d'abord installer les drivers NVIDIA.** Consultez `INSTALL_NVIDIA_DRIVERS.md` et suivez les instructions. Revenez ici une fois que `nvidia-smi` fonctionne.

## 🔍 Étape 1 : Vérifier votre GPU

```bash
lspci | grep -i nvidia
```

Ou vérifier si les drivers sont déjà installés :

```bash
nvidia-smi
```

**Si vous voyez "NVIDIA-SMI has failed" :** Les drivers ne sont pas installés. Suivez `INSTALL_NVIDIA_DRIVERS.md` d'abord.

Si `nvidia-smi` fonctionne et affiche des informations sur votre GPU, passez à l'étape 2.

## 🚀 Installation Rapide (Méthode Recommandée)

### Option A : Installation automatique avec TensorFlow

TensorFlow 2.13+ peut installer automatiquement CUDA et cuDNN via pip :

```bash
# Activer votre venv
source venv/bin/activate

# Désinstaller TensorFlow CPU si présent
pip uninstall tensorflow tensorflow-cpu

# Installer TensorFlow avec CUDA/cuDNN automatique
pip install tensorflow[and-cuda]
```

Cette méthode installe automatiquement :
- CUDA Toolkit
- cuDNN
- TensorFlow avec support GPU

**C'est la méthode la plus simple !**

### Option B : Installation manuelle de CUDA (si Option A ne fonctionne pas)

#### 1. Installer les drivers NVIDIA

```bash
# Vérifier le modèle de votre GPU
lspci | grep -i nvidia

# Installer les drivers (remplacez 535 par la version appropriée)
sudo apt update
sudo apt install nvidia-driver-535  # ou nvidia-driver-525, etc.

# Redémarrer
sudo reboot
```

Après redémarrage, vérifiez :
```bash
nvidia-smi
```

#### 2. Installer CUDA via pip (plus simple que l'installation système)

```bash
# Dans votre venv
pip install nvidia-cudnn-cu12  # Pour CUDA 12
# ou
pip install nvidia-cudnn-cu11  # Pour CUDA 11.8
```

#### 3. Installer TensorFlow GPU

```bash
pip install tensorflow[and-cuda]
```

## ✅ Vérification

```bash
python test_gpu.py
```

Vous devriez voir :
```
✓ GPU(s) trouvé(s)
✓ Configuration mémoire GPU activée
✓ Test réussi!
```

## 🆘 Dépannage

### "nvcc: commande introuvable"

**Solution :** Utilisez l'Option A ci-dessus. `pip install tensorflow[and-cuda]` installe tout automatiquement sans avoir besoin de `nvcc` dans le PATH.

### "No GPU devices found"

1. **Vérifier les drivers :**
   ```bash
   nvidia-smi
   ```
   Si ça ne fonctionne pas :
   ```bash
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

2. **Réinstaller TensorFlow :**
   ```bash
   pip uninstall tensorflow tensorflow-cpu
   pip install tensorflow[and-cuda]
   ```

3. **Vérifier dans Python :**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### "Could not load dynamic library"

Cela signifie que TensorFlow ne trouve pas les bibliothèques CUDA.

**Solution :**
```bash
# Réinstaller avec toutes les dépendances
pip uninstall tensorflow tensorflow-cpu
pip install tensorflow[and-cuda] --upgrade
```

## 📝 Commandes Rapides

```bash
# 1. Activer venv
source venv/bin/activate

# 2. Installer TensorFlow GPU (installe tout automatiquement)
pip install tensorflow[and-cuda]

# 3. Tester
python test_gpu.py

# 4. Si ça fonctionne, entraîner
python main.py 3
```

## 💡 Note Importante

Avec `pip install tensorflow[and-cuda]`, vous n'avez **PAS besoin** d'installer CUDA manuellement via `apt` ou depuis le site NVIDIA. Tout est géré automatiquement par pip dans votre environnement virtuel.
