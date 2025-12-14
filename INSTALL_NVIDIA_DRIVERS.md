# Installation des Drivers NVIDIA sur Ubuntu/Debian

Guide pour installer les drivers NVIDIA quand `nvidia-smi` ne fonctionne pas.

## 🔍 Étape 1 : Vérifier votre GPU

```bash
lspci | grep -i nvidia
```

Cela devrait afficher votre GPU NVIDIA. Notez le modèle.

## 🚀 Installation des Drivers

### Méthode 1 : Installation automatique (Recommandée)

Ubuntu peut détecter et installer automatiquement les drivers :

```bash
# Mettre à jour les paquets
sudo apt update

# Installer les outils de détection
sudo apt install ubuntu-drivers-common

# Détecter et installer automatiquement les drivers recommandés
sudo ubuntu-drivers autoinstall

# OU installer manuellement la version recommandée
ubuntu-drivers devices  # Voir les drivers disponibles
sudo apt install nvidia-driver-535  # Remplacez 535 par la version recommandée
```

### Méthode 2 : Installation via le gestionnaire de paquets

```bash
# Mettre à jour
sudo apt update

# Voir les drivers disponibles
apt search nvidia-driver

# Installer le driver (version récente, généralement 535 ou 525)
sudo apt install nvidia-driver-535

# Redémarrer
sudo reboot
```

### Méthode 3 : Via le gestionnaire graphique (Ubuntu)

1. Ouvrez **Paramètres** → **Pilotes additionnels**
2. Sélectionnez le driver NVIDIA recommandé
3. Cliquez sur **Appliquer les modifications**
4. Redémarrez

## ✅ Vérification après redémarrage

Après avoir redémarré, vérifiez :

```bash
nvidia-smi
```

Vous devriez voir des informations sur votre GPU. Si ça fonctionne, passez à l'installation de TensorFlow GPU.

## 🆘 Dépannage

### Problème : "NVIDIA-SMI has failed"

**Solutions :**

1. **Vérifier que le GPU est détecté :**
   ```bash
   lspci | grep -i nvidia
   ```

2. **Vérifier les drivers installés :**
   ```bash
   dpkg -l | grep nvidia
   ```

3. **Réinstaller les drivers :**
   ```bash
   sudo apt remove --purge '^nvidia-.*'
   sudo apt autoremove
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

### Problème : Boucle de login après installation

Si vous êtes bloqué au login après l'installation :

1. **Mode recovery :**
   - Au démarrage, maintenez Shift pour accéder au menu GRUB
   - Sélectionnez "Advanced options" → "Recovery mode"
   - Choisissez "root" ou "resume"

2. **Désinstaller les drivers :**
   ```bash
   sudo apt remove --purge '^nvidia-.*'
   sudo reboot
   ```

3. **Réessayer avec une version différente :**
   ```bash
   sudo apt install nvidia-driver-525  # Version plus ancienne
   ```

### Problème : Conflit avec nouveau noyau

Si vous avez mis à jour le noyau récemment :

```bash
# Vérifier la version du noyau
uname -r

# Réinstaller les drivers pour le nouveau noyau
sudo apt install --reinstall nvidia-driver-535
sudo reboot
```

## 📝 Commandes Utiles

```bash
# Voir les drivers NVIDIA installés
dpkg -l | grep nvidia

# Voir les modules NVIDIA chargés
lsmod | grep nvidia

# Voir les informations détaillées du GPU
lspci -v | grep -i nvidia -A 12

# Vérifier les erreurs dans les logs
dmesg | grep -i nvidia
```

## 🔄 Après Installation Réussie

Une fois que `nvidia-smi` fonctionne :

1. **Installer TensorFlow GPU :**
   ```bash
   source venv/bin/activate
   pip install tensorflow[and-cuda]
   ```

2. **Tester :**
   ```bash
   python test_gpu.py
   ```

## 💡 Note

- Les drivers NVIDIA nécessitent généralement un **redémarrage** après installation
- Si vous utilisez un laptop avec GPU hybride (Optimus), vous pourriez avoir besoin de configurations supplémentaires
- Consultez `INSTALL_GPU_UBUNTU.md` pour la suite après l'installation des drivers
