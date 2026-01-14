#!/usr/bin/env python3
"""
================================================================================
                    ANALYSE ET RÉDUCTION DES FEATURES AUDIO
================================================================================

Ce script réalise une analyse complète des features extraites des fichiers audio
pour préparer les données à l'entraînement d'un modèle de Machine Learning.

PIPELINE COMPLET :
    1. Chargement des données depuis le CSV
    2. Équilibrage des classes (optionnel) - évite le biais vers la classe majoritaire
    3. Analyse de corrélation - identifie les features redondantes
    4. Réduction des features - supprime les features trop corrélées (>0.95)
    5. Standardisation - normalise les données (moyenne=0, écart-type=1)
    6. PCA - réduit à 2 dimensions pour visualisation
    7. Random Forest - identifie les features les plus importantes

POURQUOI CES ÉTAPES ?
    - Équilibrage : Un dataset déséquilibré (ex: 90% Normal, 10% Bruyant) va
                    créer un modèle biaisé qui prédit toujours "Normal"
    - Corrélation : Deux features très corrélées apportent la même information,
                    garder les deux surcharge le modèle inutilement
    - Standardisation : Les algorithmes ML sont sensibles à l'échelle des données.
                        Une feature en [0-1000] dominerait une feature en [0-1]
    - PCA : Permet de visualiser les données en 2D et voir si les classes
            sont séparables
    - Random Forest : Identifie quelles features le modèle utilise vraiment
                      pour prendre ses décisions

Usage:
    python feature_analysis.py              # Avec équilibrage (par défaut)
    python feature_analysis.py --balance    # Avec équilibrage
    python feature_analysis.py --no-balance # Sans équilibrage (garde tout)
    python feature_analysis.py --help       # Affiche l'aide

Auteur: Daniel
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os      # Pour vérifier si les fichiers existent
import sys     # Pour lire les arguments en ligne de commande
import numpy as np      # Calculs numériques (matrices, arrays)
import pandas as pd     # Manipulation de données tabulaires (CSV)
import seaborn as sns   # Visualisations statistiques élégantes
import matplotlib.pyplot as plt  # Création de graphiques

# Scikit-learn : La bibliothèque de Machine Learning de référence en Python
from sklearn.preprocessing import StandardScaler    # Normalisation des données
from sklearn.decomposition import PCA               # Réduction de dimensionnalité
from sklearn.ensemble import RandomForestClassifier # Modèle de classification
from sklearn.utils import resample                  # Sous-échantillonnage

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Chemins des fichiers d'entrée et de sortie
FEATURES_CSV = "data/features.csv"       # Fichier d'entrée avec toutes les features
OUTPUT_CSV = "data/features_reduced.csv" # Fichier de sortie avec features nettoyées

# ==============================================================================
# COULEURS ANSI POUR L'AFFICHAGE TERMINAL
# ==============================================================================
# Les codes ANSI permettent d'afficher du texte coloré dans le terminal.
# Format : \033[XXm où XX est un code de couleur
# \033[0m (END) remet la couleur par défaut

class Colors:
    HEADER = '\033[95m'   # Magenta clair
    BLUE = '\033[94m'     # Bleu
    CYAN = '\033[96m'     # Cyan
    GREEN = '\033[92m'    # Vert
    YELLOW = '\033[93m'   # Jaune
    RED = '\033[91m'      # Rouge
    BOLD = '\033[1m'      # Gras
    DIM = '\033[2m'       # Atténué
    END = '\033[0m'       # Reset (fin de formatage)


# ==============================================================================
# FONCTIONS D'AFFICHAGE
# ==============================================================================
# Ces fonctions facilitent l'affichage formaté dans le terminal

def print_header(title: str):
    """
    Affiche un titre de section encadré.
    
    Exemple de sortie:
    ──────────────────────────────────────────────────
      TITRE DE LA SECTION
    ──────────────────────────────────────────────────
    """
    width = 50
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'─' * width}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {title.upper()}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * width}{Colors.END}")


def print_success(msg: str):
    """Affiche un message de succès en vert avec [OK]"""
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")


def print_info(msg: str):
    """Affiche un message d'information en bleu avec [i]"""
    print(f"{Colors.BLUE}[i]{Colors.END} {msg}")


def print_warning(msg: str):
    """Affiche un message d'avertissement en jaune avec [!]"""
    print(f"{Colors.YELLOW}[!]{Colors.END} {msg}")


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main(balance: bool = True):
    """
    Fonction principale qui exécute tout le pipeline d'analyse.
    
    Paramètres:
    -----------
    balance : bool (défaut=True)
        Si True, équilibre les classes pour avoir 50/50.
        Si False, garde toutes les données même si déséquilibrées.
    
    Retourne:
    ---------
    X_scaled : np.array
        Les features standardisées (prêtes pour ML)
    feature_names : list
        Liste des noms des features conservées
    labels : pd.Series
        Les labels (1=Bruyant, 2=Normal)
    pca : PCA
        L'objet PCA entraîné (pour transformer de nouvelles données)
    loadings : pd.DataFrame
        Les poids de chaque feature dans les composantes PCA
    """
    
    # ==========================================================================
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # ==========================================================================
    # On charge le fichier CSV qui contient toutes les features extraites
    # des fichiers audio par feature_extraction.py
    
    print_header("Chargement des données")
    
    # Vérifier que le fichier existe avant de le charger
    if not os.path.exists(FEATURES_CSV):
        print(f"{Colors.RED}[ERREUR]{Colors.END} Fichier non trouvé: {FEATURES_CSV}")
        return
    
    # Charger le CSV dans un DataFrame pandas
    # Un DataFrame est comme un tableau Excel : lignes = échantillons, colonnes = features
    df = pd.read_csv(FEATURES_CSV)
    
    print_info(f"{len(df)} échantillons chargés")
    print_info(f"{len(df.columns)} colonnes totales")
    
    # ==========================================================================
    # ÉTAPE 2 : ÉQUILIBRAGE DES CLASSES
    # ==========================================================================
    # POURQUOI ÉQUILIBRER ?
    # Si on a 200 "Normal" et 20 "Bruyant", le modèle va apprendre à toujours
    # prédire "Normal" car c'est correct 90% du temps !
    # En équilibrant (50/50), on force le modèle à vraiment apprendre les
    # différences entre les deux classes.
    #
    # TECHNIQUE : Sous-échantillonnage (downsampling)
    # On réduit la classe majoritaire pour égaler la classe minoritaire.
    # Alternative : Sur-échantillonnage (upsampling) = dupliquer la minoritaire
    
    print_header("Équilibrage des classes")
    
    if 'label' in df.columns:
        # Compter combien d'échantillons par classe
        # value_counts() retourne un dictionnaire {label: count}
        class_counts = df['label'].value_counts()
        
        # Afficher la distribution actuelle avec une barre visuelle
        print(f"\n  {Colors.BOLD}Distribution actuelle:{Colors.END}")
        for label, count in class_counts.items():
            # Mapper les labels numériques vers des noms lisibles
            label_name = "Bruyant" if label == 1 else "Normal"
            # Calculer le pourcentage
            pct = count / len(df) * 100
            # Créer une barre de progression visuelle (chaque █ = 5%)
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"    Label {label} ({label_name}): {bar} {count:>4} ({pct:.1f}%)")
        
        # Calculer le ratio entre la plus petite et la plus grande classe
        # Un ratio de 1.0 = parfaitement équilibré
        # Un ratio de 0.1 = très déséquilibré (10x plus d'une classe)
        min_class = class_counts.min()  # Nombre d'échantillons de la classe minoritaire
        max_class = class_counts.max()  # Nombre d'échantillons de la classe majoritaire
        ratio = min_class / max_class   # Ratio entre les deux
        
        # Décider si on équilibre ou non
        if not balance:
            # L'utilisateur a demandé de ne pas équilibrer
            print()
            print_info("Équilibrage désactivé (--no-balance)")
            print_warning(f"Ratio actuel: {ratio:.1%}")
            
        elif ratio < 0.8:  # On considère déséquilibré si ratio < 80%
            print()
            print_warning(f"Déséquilibre détecté (ratio {ratio:.1%})")
            print_info("Application du sous-échantillonnage...")
            
            # Séparer le DataFrame par classe
            # Chaque groupe contient tous les échantillons d'une classe
            df_groups = [df[df['label'] == label] for label in class_counts.index]
            
            # On va sous-échantillonner pour avoir autant d'échantillons
            # que la classe minoritaire
            n_samples = min_class
            df_balanced_groups = []
            
            for group in df_groups:
                if len(group) > n_samples:
                    # Cette classe a trop d'échantillons, on en prend un sous-ensemble
                    # resample() tire aléatoirement n_samples échantillons
                    # replace=False : pas de doublons (chaque échantillon pris une seule fois)
                    # random_state=42 : graine pour reproductibilité
                    group_downsampled = resample(
                        group,
                        replace=False,
                        n_samples=n_samples,
                        random_state=42
                    )
                    df_balanced_groups.append(group_downsampled)
                else:
                    # Cette classe a le bon nombre ou moins, on garde tout
                    df_balanced_groups.append(group)
            
            # Recombiner tous les groupes en un seul DataFrame
            df = pd.concat(df_balanced_groups)
            
            # Mélanger les lignes aléatoirement
            # Important : sinon tous les "Bruyant" seraient en premier !
            # frac=1 = garder 100% des données (juste mélanger)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Afficher la nouvelle distribution
            print()
            print(f"  {Colors.BOLD}Après équilibrage:{Colors.END}")
            new_counts = df['label'].value_counts()
            for label, count in new_counts.items():
                label_name = "Bruyant" if label == 1 else "Normal"
                pct = count / len(df) * 100
                print(f"    Label {label} ({label_name}): {count:>4} ({pct:.1f}%)")
            
            print_success(f"Dataset équilibré: {len(df)} échantillons (50/50)")
        else:
            # Le dataset est déjà équilibré (ratio >= 80%)
            print()
            print_success(f"Dataset déjà équilibré (ratio {ratio:.1%})")
    else:
        print_warning("Pas de colonne 'label' - équilibrage ignoré")
    
    # ==========================================================================
    # SÉPARATION FEATURES / MÉTADONNÉES
    # ==========================================================================
    # Le CSV contient des colonnes qui ne sont pas des features :
    # - nfile : nom du fichier audio
    # - label : la classe (1=Bruyant, 2=Normal)
    # - reliability : score de fiabilité de l'annotation
    # On doit les séparer pour ne garder que les features numériques
    
    metadata_cols = ['nfile', 'label', 'reliability', 'duration']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # X = matrice des features (ce que le modèle va analyser)
    # labels = vecteur des classes (ce que le modèle doit prédire)
    X = df[feature_cols].copy()
    labels = df['label'] if 'label' in df.columns else None
    
    print()
    print_info(f"{len(feature_cols)} features numériques")
    if labels is not None:
        print_info(f"Labels: {sorted(labels.unique())}")
    
    # ==========================================================================
    # ÉTAPE 3 : ANALYSE DE CORRÉLATION
    # ==========================================================================
    # POURQUOI ANALYSER LA CORRÉLATION ?
    # Deux features très corrélées (ex: corrélation > 0.95) apportent
    # la même information. Garder les deux :
    #   - Surcharge le modèle inutilement
    #   - Peut créer de la multicolinéarité (problème pour certains algorithmes)
    #   - Ralentit l'entraînement
    #
    # La matrice de corrélation montre la corrélation entre chaque paire de features.
    # Valeurs :
    #   - 1.0 = parfaitement corrélées (identiques)
    #   - 0.0 = aucune corrélation
    #   - -1.0 = parfaitement anti-corrélées (opposées)
    # On prend la valeur absolue car -0.95 est aussi "très corrélé" que +0.95
    
    print_header("Analyse de corrélation")
    
    # Calculer la matrice de corrélation
    # .corr() calcule la corrélation de Pearson entre chaque paire de colonnes
    # .abs() prend la valeur absolue
    corr_matrix = X.corr().abs()
    
    # Pour compter les paires corrélées, on ne regarde que le triangle supérieur
    # de la matrice (sinon on compte chaque paire deux fois)
    # np.triu() crée un masque pour le triangle supérieur
    # k=1 exclut la diagonale (une feature est toujours corrélée à 1.0 avec elle-même)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Compter les paires fortement et moyennement corrélées
    high_corr = (upper > 0.95).sum().sum()      # Très corrélées (à supprimer)
    medium_corr = ((upper > 0.8) & (upper <= 0.95)).sum().sum()  # Moyennement corrélées
    
    print_info(f"Paires fortement corrélées (>0.95): {high_corr}")
    print_info(f"Paires moyennement corrélées (0.8-0.95): {medium_corr}")
    
    # --------------------------------------------------------------------------
    # VISUALISATION : HEATMAP DE CORRÉLATION
    # --------------------------------------------------------------------------
    # Une heatmap (carte de chaleur) permet de visualiser la matrice de corrélation.
    # Couleurs : bleu = faible corrélation, rouge = forte corrélation
    
    print()
    print(f"  {Colors.DIM}Génération de la heatmap...{Colors.END}")
    
    # Calculer la taille de la figure en fonction du nombre de features
    # Plus il y a de features, plus la figure doit être grande
    n_features = len(corr_matrix.columns)
    fig_size = n_features * 0.35  # 0.35 pouces par feature
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Palette de couleurs divergente : bleu -> blanc -> rouge
    # 220 = teinte bleue, 20 = teinte rouge
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Créer la heatmap avec seaborn
    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmin=0, vmax=1,      # Échelle de 0 à 1 (valeurs absolues)
        annot=False,         # Pas d'annotations (trop de cellules)
        square=True,         # Cellules carrées
        linewidths=1,        # Lignes blanches entre les cellules
        linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': 'Corrélation'},  # Barre de couleur
        ax=ax
    )
    
    # Rotation des labels pour la lisibilité
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    ax.set_title("Matrice de Corrélation des Features Audio",
                 fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print_success("Heatmap sauvegardée: data/correlation_matrix.png")
    plt.show()
    
    # ==========================================================================
    # ÉTAPE 4 : SUPPRESSION DES FEATURES REDONDANTES
    # ==========================================================================
    # On supprime les features qui ont une corrélation > 0.95 avec une autre.
    # Pour chaque paire très corrélée, on garde une et on supprime l'autre.
    
    print_header("Réduction des features")
    
    # Trouver les colonnes à supprimer
    # Pour chaque colonne, on vérifie si elle est corrélée à > 0.95 avec une autre
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        print_warning(f"{len(to_drop)} colonnes à supprimer (corrélation > 0.95):")
        # Afficher les 10 premières colonnes à supprimer
        for col in to_drop[:10]:
            print(f"    {Colors.DIM}• {col}{Colors.END}")
        if len(to_drop) > 10:
            print(f"    {Colors.DIM}... et {len(to_drop) - 10} autres{Colors.END}")
    
    # Créer le nouveau DataFrame sans les colonnes redondantes
    X_reduced = X.drop(columns=to_drop)
    
    print()
    print_info(f"Features initiales: {X.shape[1]}")
    print_success(f"Features restantes: {X_reduced.shape[1]}")
    print_info(f"Réduction: {X.shape[1] - X_reduced.shape[1]} colonnes supprimées ({(len(to_drop)/X.shape[1])*100:.1f}%)")
    
    # ==========================================================================
    # ÉTAPE 5 : STANDARDISATION
    # ==========================================================================
    # POURQUOI STANDARDISER ?
    # Les features ont des échelles très différentes :
    #   - spectral_centroid : valeurs autour de 1000-5000 Hz
    #   - mfcc : valeurs autour de -300 à +150
    #   - energy : valeurs autour de 0.01 à 1
    #
    # Sans standardisation, les features avec les plus grandes valeurs
    # dominent les calculs de distance et faussent les algorithmes.
    #
    # FORMULE : z = (x - μ) / σ
    # Où μ = moyenne de la feature, σ = écart-type de la feature
    # Résultat : chaque feature a moyenne = 0 et écart-type = 1
    
    print_header("Standardisation")
    
    # StandardScaler : outil de scikit-learn pour standardiser
    scaler = StandardScaler()
    
    # fit_transform() fait deux choses :
    #   1. fit() : calcule μ et σ pour chaque feature
    #   2. transform() : applique la formule z = (x - μ) / σ
    X_scaled = scaler.fit_transform(X_reduced)
    
    print_info("Moyenne → 0, Écart-type → 1")
    print_success("Données prêtes pour PCA/ML")
    
    # Reconvertir en DataFrame pour garder les noms de colonnes
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_reduced.columns)
    
    # Réajouter les métadonnées (label, nfile, reliability)
    # Ces colonnes n'ont pas été standardisées car ce ne sont pas des features
    if labels is not None:
        X_scaled_df['label'] = labels.values
    if 'nfile' in df.columns:
        X_scaled_df['nfile'] = df['nfile'].values
    if 'reliability' in df.columns:
        X_scaled_df['reliability'] = df['reliability'].values
    
    # Sauvegarder le dataset nettoyé et standardisé
    X_scaled_df.to_csv(OUTPUT_CSV, index=False)
    print_success(f"Données sauvegardées: {OUTPUT_CSV}")
    
    # ==========================================================================
    # ÉTAPE 6 : PCA (Analyse en Composantes Principales)
    # ==========================================================================
    # POURQUOI LA PCA ?
    # On a ~100 features, impossible à visualiser ! La PCA permet de réduire
    # à 2 dimensions tout en gardant un maximum d'information.
    #
    # COMMENT ÇA MARCHE ?
    # La PCA trouve les "directions" (composantes) qui capturent le plus
    # de variance dans les données.
    #   - PC1 (1ère composante) : direction de variance maximale
    #   - PC2 (2ème composante) : direction de 2ème plus grande variance,
    #                            perpendiculaire à PC1
    #
    # Si PC1 capture 40% de la variance et PC2 30%, alors le plot 2D
    # capture 70% de l'information totale !
    #
    # INTERPRÉTATION DU PLOT :
    #   - Points proches = échantillons similaires
    #   - Points éloignés = échantillons différents
    #   - Clusters séparés par classe = bon signe (les features distinguent bien)
    
    print_header("PCA - Analyse en Composantes Principales")
    
    # Créer l'objet PCA pour réduire à 2 dimensions
    pca = PCA(n_components=2)
    
    # fit_transform() :
    #   1. fit() : calcule les composantes principales
    #   2. transform() : projette les données sur ces composantes
    principal_components = pca.fit_transform(X_scaled)
    
    # Récupérer la variance expliquée par chaque composante
    # explained_variance_ratio_ = proportion de variance capturée
    var1 = pca.explained_variance_ratio_[0]  # Variance de PC1
    var2 = pca.explained_variance_ratio_[1]  # Variance de PC2
    var_total = var1 + var2                   # Variance totale capturée
    
    print_info(f"PC1: {var1:.1%} de variance expliquée")
    print_info(f"PC2: {var2:.1%} de variance expliquée")
    print_success(f"Total: {var_total:.1%} de variance capturée en 2D")
    
    # Créer un DataFrame pour le scatter plot
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # Ajouter les labels pour colorer les points par classe
    if labels is not None:
        # Mapper les labels numériques vers des noms lisibles
        label_map = {1: 'Bruyant', 2: 'Normal'}
        df_pca['target'] = labels.map(label_map).fillna('Inconnu')
    else:
        df_pca['target'] = 'Inconnu'
    
    # --------------------------------------------------------------------------
    # VISUALISATION : SCATTER PLOT PCA
    # --------------------------------------------------------------------------
    print()
    print(f"  {Colors.DIM}Génération du scatter plot PCA...{Colors.END}")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#fafafa')  # Fond gris très clair
    
    # Couleurs et marqueurs pour chaque classe
    palette = {'Bruyant': '#e63946', 'Normal': '#2a9d8f', 'Inconnu': '#adb5bd'}
    markers = {'Bruyant': 'X', 'Normal': 'o', 'Inconnu': 's'}
    
    # Tracer chaque classe séparément (pour avoir des marqueurs différents)
    for target in df_pca['target'].unique():
        subset = df_pca[df_pca['target'] == target]
        ax.scatter(
            subset['PC1'], subset['PC2'],
            c=palette.get(target, '#adb5bd'),  # Couleur
            marker=markers.get(target, 'o'),    # Forme du point
            s=120,              # Taille des points
            alpha=0.75,         # Transparence
            edgecolor='white',  # Contour blanc
            linewidth=1.5,
            label=target,
            zorder=3            # Au-dessus de la grille
        )
    
    # Grille et axes
    ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc', zorder=1)
    ax.axhline(y=0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(x=0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5, zorder=2)
    
    # Labels des axes (avec pourcentage de variance)
    ax.set_xlabel(f'PC1  ({var1:.1%} de variance)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'PC2  ({var2:.1%} de variance)', fontsize=12, fontweight='bold', labelpad=10)
    
    ax.set_title('Projection PCA des Échantillons Audio',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Sous-titre avec variance totale
    fig.text(0.5, 0.91, f'Variance totale capturée: {var_total:.1%}',
             ha='center', fontsize=11, color='#666666', style='italic')
    
    # Légende stylisée
    legend = ax.legend(
        title='Classification',
        title_fontsize=11,
        fontsize=10,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95
    )
    legend.get_frame().set_facecolor('white')
    
    # Badge avec le nombre d'échantillons
    stats_text = f"n = {len(df_pca)} échantillons"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('data/pca_visualization.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print_success("PCA plot sauvegardé: data/pca_visualization.png")
    plt.show()
    
    # ==========================================================================
    # ÉTAPE 7 : LOADINGS PCA (Poids des features)
    # ==========================================================================
    # Les "loadings" indiquent combien chaque feature contribue à chaque
    # composante principale.
    # Un loading élevé (positif ou négatif) = cette feature est importante
    # pour cette composante.
    #
    # UTILITÉ : Comprendre ce que "regarde" la PCA
    # Si PC1 sépare bien Bruyant/Normal, les features avec les plus gros
    # loadings sur PC1 sont celles qui distinguent le mieux les deux classes.
    
    print_header("Features les plus influentes")
    
    # pca.components_ : matrice [n_composantes x n_features]
    # On transpose pour avoir [n_features x n_composantes]
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=X_reduced.columns
    )
    
    # Afficher le Top 5 des features influentes sur PC1
    print(f"\n  {Colors.BOLD}Top 5 - Axe horizontal (PC1):{Colors.END}")
    top_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(5)
    for i, (feat, val) in enumerate(top_pc1.items(), 1):
        sign = '+' if loadings.loc[feat, 'PC1'] > 0 else '-'
        bar = '█' * int(val * 20)
        print(f"    {i}. {feat:<30} {sign}{val:.3f}  {Colors.CYAN}{bar}{Colors.END}")
    
    # Afficher le Top 5 des features influentes sur PC2
    print(f"\n  {Colors.BOLD}Top 5 - Axe vertical (PC2):{Colors.END}")
    top_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(5)
    for i, (feat, val) in enumerate(top_pc2.items(), 1):
        sign = '+' if loadings.loc[feat, 'PC2'] > 0 else '-'
        bar = '█' * int(val * 20)
        print(f"    {i}. {feat:<30} {sign}{val:.3f}  {Colors.CYAN}{bar}{Colors.END}")
    
    # Sauvegarder les loadings
    loadings.to_csv('data/pca_loadings.csv')
    print()
    print_success("Loadings sauvegardés: data/pca_loadings.csv")
    
    # ==========================================================================
    # ÉTAPE 8 : RANDOM FOREST - FEATURE IMPORTANCE
    # ==========================================================================
    # POURQUOI LE RANDOM FOREST ?
    # C'est un algorithme de Machine Learning qui peut nous dire quelles
    # features sont les plus utiles pour distinguer Bruyant de Normal.
    #
    # COMMENT ÇA MARCHE ?
    # Un Random Forest est composé de nombreux arbres de décision.
    # Chaque arbre "vote" pour une classe. La classe majoritaire gagne.
    # La "feature importance" mesure combien chaque feature contribue
    # à la bonne classification.
    #
    # UTILITÉ :
    # - Savoir quelles features garder pour un modèle léger
    # - Comprendre ce qui distingue Bruyant de Normal
    # - Valider les résultats de la PCA
    
    print_header("Random Forest - Importance des Features")
    
    # On a besoin d'au moins 2 classes différentes pour entraîner
    if labels is not None and len(labels.unique()) >= 2:
        print_info("Entraînement du Random Forest...")
        
        # y = vecteur des labels (ce qu'on veut prédire)
        y = labels.values
        
        # Créer et entraîner le Random Forest
        # n_estimators=100 : 100 arbres dans la forêt
        # random_state=42 : graine pour reproductibilité
        # n_jobs=-1 : utiliser tous les CPU disponibles
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)  # Entraîner sur les données standardisées
        
        # Récupérer l'importance de chaque feature
        # feature_importances_ : array avec l'importance de chaque feature
        # La somme fait 1.0 (100%)
        importances = rf.feature_importances_
        feature_names = X_reduced.columns
        
        # Créer un DataFrame trié par importance décroissante
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
        
        # Afficher le Top 15 dans le terminal
        print()
        print(f"  {Colors.BOLD}Top 15 des paramètres déterminants:{Colors.END}")
        for i, (_, row) in enumerate(feature_imp_df.head(15).iterrows(), 1):
            # Créer une barre de progression visuelle
            bar_len = int(row['Importance'] * 100)
            bar = '█' * bar_len + '░' * (15 - bar_len)
            print(f"    {i:>2}. {row['Feature']:<30} {bar} {row['Importance']:.3f}")
        
        # ----------------------------------------------------------------------
        # VISUALISATION : BARPLOT DE L'IMPORTANCE
        # ----------------------------------------------------------------------
        print()
        print(f"  {Colors.DIM}Génération du graphique...{Colors.END}")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        top_15 = feature_imp_df.head(15)
        # Dégradé de couleurs viridis (vert -> jaune)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 15))
        
        # Barres horizontales
        bars = ax.barh(
            range(len(top_15)),
            top_15['Importance'].values,
            color=colors[::-1],  # Inverser pour que le plus important soit le plus coloré
            edgecolor='white',
            linewidth=0.5
        )
        
        # Labels des barres
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels(top_15['Feature'].values, fontsize=11)
        ax.invert_yaxis()  # Plus important en haut
        
        ax.set_xlabel("Score d'importance", fontsize=12, fontweight='bold')
        ax.set_title(
            "LES 15 PARAMÈTRES QUI DÉTERMINENT LE BRUIT\n(Random Forest Feature Importance)",
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Ajouter les valeurs à droite de chaque barre
        for i, (bar, val) in enumerate(zip(bars, top_15['Importance'].values)):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, color='#333333')
        
        ax.set_xlim(0, top_15['Importance'].max() * 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)  # Grille derrière les barres
        
        plt.tight_layout()
        plt.savefig('data/feature_importance.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print_success("Graphique sauvegardé: data/feature_importance.png")
        plt.show()
        
        # Afficher le Top 5 (pour un modèle léger)
        print()
        print(f"  {Colors.BOLD}{Colors.GREEN}► Top 5 pour un modèle léger:{Colors.END}")
        top5 = feature_imp_df['Feature'].head(5).values
        for i, feat in enumerate(top5, 1):
            print(f"    {Colors.GREEN}{i}. {feat}{Colors.END}")
        
        # Sauvegarder l'importance complète
        feature_imp_df.to_csv('data/feature_importance.csv', index=False)
        print()
        print_success("Importance sauvegardée: data/feature_importance.csv")
        
    else:
        print_warning("Labels insuffisants pour Random Forest (besoin de 2+ classes)")
    
    # ==========================================================================
    # ÉTAPE 9 : RÉSUMÉ FINAL
    # ==========================================================================
    
    print_header("Résumé")
    
    print(f"""
  {Colors.BOLD}Pipeline complet:{Colors.END}
  
  1. Features initiales:     {X.shape[1]}
  2. Après dé-corrélation:   {X_reduced.shape[1]} ({Colors.GREEN}-{len(to_drop)} redondantes{Colors.END})
  3. Standardisation:        ✓ (μ=0, σ=1)
  4. PCA 2D:                 {var_total:.1%} variance capturée
  5. Random Forest:          ✓ Feature importance calculée
  
  {Colors.BOLD}Fichiers générés:{Colors.END}
  • data/correlation_matrix.png   - Heatmap des corrélations
  • data/pca_visualization.png    - Scatter plot PCA 2D
  • data/feature_importance.png   - Barplot importance features
  • data/features_reduced.csv     - Dataset nettoyé et standardisé
  • data/pca_loadings.csv         - Poids des features dans la PCA
  • data/feature_importance.csv   - Scores d'importance Random Forest
  
  {Colors.GREEN}✓{Colors.END} Analyse terminée !
    """)
    
    return X_scaled, X_reduced.columns.tolist(), labels, pca, loadings


# ==============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ==============================================================================
# Ce bloc s'exécute uniquement quand on lance le script directement
# (pas quand on l'importe comme module)

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # PARSING DES ARGUMENTS
    # --------------------------------------------------------------------------
    # On regarde si l'utilisateur a passé des arguments en ligne de commande
    
    balance = True  # Par défaut : équilibrage activé
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()  # Premier argument, en minuscules
        
        if arg in ['--no-balance', '-nb', 'no-balance', 'nobalance']:
            # L'utilisateur veut désactiver l'équilibrage
            balance = False
            
        elif arg in ['--balance', '-b', 'balance']:
            # L'utilisateur veut activer l'équilibrage (déjà activé par défaut)
            balance = True
            
        elif arg in ['--help', '-h', 'help']:
            # L'utilisateur demande l'aide
            print(f"""
{Colors.BOLD}Feature Analysis - Analyse des features audio{Colors.END}

{Colors.CYAN}Usage:{Colors.END}
    python feature_analysis.py              # Avec équilibrage (défaut)
    python feature_analysis.py --balance    # Avec équilibrage
    python feature_analysis.py --no-balance # Sans équilibrage

{Colors.CYAN}Options:{Colors.END}
    --balance, -b       Équilibre les classes (50/50)
    --no-balance, -nb   Garde toutes les données
    --help, -h          Affiche cette aide

{Colors.CYAN}Fichiers générés:{Colors.END}
    data/correlation_matrix.png   Heatmap des corrélations
    data/pca_visualization.png    Projection PCA 2D
    data/feature_importance.png   Importance des features
    data/features_reduced.csv     Dataset nettoyé
    data/pca_loadings.csv         Loadings PCA
    data/feature_importance.csv   Scores d'importance
            """)
            sys.exit(0)
    
    # --------------------------------------------------------------------------
    # EXÉCUTION
    # --------------------------------------------------------------------------
    try:
        main(balance=balance)
    except KeyboardInterrupt:
        # L'utilisateur a appuyé sur Ctrl+C
        print(f"\n{Colors.YELLOW}[!]{Colors.END} Annulé par l'utilisateur")
    except Exception as e:
        # Une erreur s'est produite
        print(f"{Colors.RED}[ERREUR]{Colors.END} {e}")
        raise  # Relancer l'exception pour voir le traceback complet
