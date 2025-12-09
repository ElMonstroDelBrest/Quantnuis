import pandas as pd

raw_data = """
| Start | End | Label | Reliability |
| 00:09:34 | 00:10:12 | 1 | 3 |
| 00:11:30 | 00:11:43 | 2 | 3 |
| 00:12:09 | 00:12:24 | 3 | 1 |
| 00:13:33 | 00:13:44 | 2 | 3 |
| 00:14:10 | 00:14:40 | 2 | 3 |
| 00:15:52 | 00:16:14 | 2 | 3 |
| 00:17:23 | 00:17:40 | 2 | 3 |
| 00:20:44 | 00:21:15 | 3 | 2 |
| 00:21:49 | 00:22:21 | 4 | 2 |
| 00:25:51 | 00:26:07 | 2 | 3 |
| 00:26:26 | 00:26:41 | 2 | 3 |
| 00:26:53 | 00:27:16 | 2 | 3 |
| 00:27:43 | 00:28:08 | 2 | 3 |
| 00:31:45 | 00:32:16 | 2 | 3 |
| 00:33:40 | 00:34:03 | 2 | 3 |
| 00:36:59 | 00:37:23 | 2 | 3 |
| 00:41:12 | 00:41:34 | 2 | 3 |
| 00:42:16 | 00:42:43 | 1 | 1 |
| 00:44:45 | 00:45:09 | 2 | 3 |
| 00:46:01 | 00:46:37 | 2 | 3 |
| 00:48:34 | 00:48:57 | 2 | 3 |
| 00:52:16 | 00:52:43 | 2 | 3 |
| 00:52:47 | 00:53:15 | 2 | 3 |
| 00:54:18 | 00:54:52 | 2 | 3 |
| 00:56:38 | 00:57:10 | 2 | 3 |
| 00:57:32 | 00:57:58 | 2 | 3 |
| 00:58:20 | 00:58:42 | 2 | 3 |
"""

# DataFrame des données
rows = []
# On nettoie les sauts de ligne et les balises éventuelles (<br>)
lines = [line.strip().replace('<br>', '') for line in raw_data.strip().split('\n')]

for line in lines:
    if not line: continue
    # On retire les barres verticales vides aux extrémités et on sépare
    parts = [p.strip() for p in line.split('|') if p.strip()]
    
    # On ignore la ligne d'en-tête ou les lignes malformées
    if len(parts) >= 4 and parts[0] not in ['Start', 'Time code']:
        # Structure attendue dans parts: [Start, End, Label, Reliability]
        rows.append(parts[:4])

# DataFrame des données
df = pd.DataFrame(rows, columns=['Start', 'End', 'Label', 'Reliability'])
df.set_index('Start', inplace=True)

# Enregistrement du DataFrame dans un fichier CSV
df.to_csv('annotations_raw.csv', index=True)
print("Données enregistrées dans 'annotations_raw.csv'")