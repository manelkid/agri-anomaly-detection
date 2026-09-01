
"""
Génère le fichier JSON pour le frontend de comparaison
À partir des résultats GEE + Copernicus + API

Sources :
- Beja_resultats_complets.csv
- copernicus_donnees_completes.csv
- comparaison_gee_copernicus.csv

Sortie :
- ../frontend/data/resultats.json
"""

import pandas as pd
import json
from pathlib import Path
import math


# ============================================================
# CONFIGURATION
# ============================================================

RESULTATS_GEE_CSV = "Beja_resultats_complets.csv"
COPERNICUS_CSV = "copernicus_donnees_completes.csv"
COMPARAISON_CSV = "comparaison_gee_copernicus.csv"

OUTPUT_DIR = Path("../frontend/data")
OUTPUT_FILE = OUTPUT_DIR / "resultats.json"


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def convertir_bool(val):
    """
    Convertit correctement différentes représentations
    en véritable booléen Python.

    Important :
        bool("False") == True en Python !

    Cette fonction évite donc ce problème.
    """

    if pd.isna(val):
        return False

    if isinstance(val, bool):
        return val

    # Cas numérique
    if isinstance(val, (int, float)):
        return val != 0

    valeur = str(val).strip().lower()

    if valeur in ["true", "1", "yes", "oui", "vrai"]:
        return True

    if valeur in ["false", "0", "no", "non", "faux", ""]:
        return False

    # Valeur inconnue -> False par sécurité
    return False


def valeur_float(val, default=0.0):
    """
    Convertit une valeur en float sans laisser de NaN
    dans le JSON.
    """

    try:
        valeur = float(val)

        if math.isnan(valeur) or math.isinf(valeur):
            return default

        return valeur

    except (ValueError, TypeError):
        return default


def valeur_str(val, default=""):
    """
    Convertit une valeur en chaîne.
    """

    if pd.isna(val):
        return default

    return str(val)


# ============================================================
# DÉBUT
# ============================================================

print("=" * 60)
print("GÉNÉRATION DES DONNÉES POUR LE FRONTEND")
print("=" * 60)

print("\n📂 Chargement des fichiers...")


# ============================================================
# 1. RÉSULTATS GEE
# ============================================================

try:

    df_gee = pd.read_csv(RESULTATS_GEE_CSV)

    print(f"  📄 Fichier GEE chargé : {RESULTATS_GEE_CSV}")

    colonnes_gee = [
        'parcel_id',
        'anomalie',
        'z_score',
        'type_anomalie',
        'criticite',
        'criticite_pct'
    ]

    # Vérification des colonnes
    colonnes_manquantes = [
        col for col in colonnes_gee
        if col not in df_gee.columns
    ]

    if colonnes_manquantes:

        print(
            f"  ❌ Colonnes GEE manquantes : "
            f"{colonnes_manquantes}"
        )

        print(
            f"  📋 Colonnes disponibles : "
            f"{df_gee.columns.tolist()}"
        )

        exit(1)

    # Conversion explicite du booléen anomalie
    df_gee['anomalie'] = df_gee['anomalie'].apply(convertir_bool)

    # Garder uniquement les colonnes nécessaires
    df_gee = df_gee[colonnes_gee].drop_duplicates(
        'parcel_id'
    )

    # Renommer pour le frontend
    df_gee.columns = [
        'parcel_id',
        'gee_anomalie',
        'gee_z_score',
        'gee_type',
        'gee_criticite',
        'gee_criticite_pct'
    ]

    print(f"  ✅ GEE : {len(df_gee)} parcelles")

    print(
        f"     Anomalies GEE : "
        f"{df_gee['gee_anomalie'].sum()}"
    )

    print(
        f"     Normales GEE : "
        f"{(~df_gee['gee_anomalie']).sum()}"
    )

except FileNotFoundError:

    print(
        f"  ❌ Fichier GEE introuvable : "
        f"{RESULTATS_GEE_CSV}"
    )

    exit(1)

except Exception as e:

    print(f"  ❌ Erreur chargement GEE : {e}")

    exit(1)


# ============================================================
# 2. COMPARAISON GEE / COPERNICUS
# ============================================================

try:

    df_comp = pd.read_csv(COMPARAISON_CSV)

    print(
        f"  📄 Fichier comparaison chargé : "
        f"{COMPARAISON_CSV}"
    )

    # --------------------------------------------------------
    # Normalisation des noms de colonnes
    # --------------------------------------------------------

    df_comp = df_comp.rename(columns={
        'anomalie_Copernicus': 'api_anomalie',
        'cause_Copernicus': 'api_type',
        'z_Copernicus': 'api_z_score',
        'ACCORD': 'accord'
    })

    print(f"  ✅ Comparaison : {len(df_comp)} parcelles")

    print("  📋 Colonnes comparaison :")
    print(f"     {df_comp.columns.tolist()}")

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    colonnes_comp_requises = [
        'parcel_id',
        'api_anomalie',
        'api_type',
        'api_z_score',
        'accord'
    ]

    colonnes_manquantes = [
        col
        for col in colonnes_comp_requises
        if col not in df_comp.columns
    ]

    if colonnes_manquantes:

        print(
            f"  ❌ Colonnes comparaison manquantes : "
            f"{colonnes_manquantes}"
        )

        exit(1)

    # --------------------------------------------------------
    # Conversion correcte des booléens
    # --------------------------------------------------------

    df_comp['accord'] = df_comp['accord'].apply(
        convertir_bool
    )

    df_comp['api_anomalie'] = df_comp['api_anomalie'].apply(
        convertir_bool
    )

    # --------------------------------------------------------
    # Vérification très importante
    # --------------------------------------------------------

    nb_accord_csv = int(df_comp['accord'].sum())

    nb_divergence_csv = int(
        (~df_comp['accord']).sum()
    )

    total_csv = len(df_comp)

    taux_csv = (
        100 * nb_accord_csv / total_csv
        if total_csv > 0
        else 0
    )

    print("\n  🔍 VÉRIFICATION DU CSV DE COMPARAISON")
    print("  " + "-" * 45)

    print(
        f"  Total       : {total_csv}"
    )

    print(
        f"  Accord      : {nb_accord_csv}"
        f" ({taux_csv:.1f}%)"
    )

    print(
        f"  Divergence  : {nb_divergence_csv}"
        f" ({100 - taux_csv:.1f}%)"
    )

    print("\n  Valeurs de la colonne 'accord' :")

    print(
        df_comp['accord'].value_counts()
    )

except FileNotFoundError:

    print(
        f"  ❌ Fichier comparaison introuvable : "
        f"{COMPARAISON_CSV}"
    )

    exit(1)

except Exception as e:

    print(
        f"  ❌ Erreur chargement comparaison : {e}"
    )

    exit(1)


# ============================================================
# 3. COORDONNÉES
# ============================================================

try:

    df_coords = pd.read_csv(
        COPERNICUS_CSV
    )[['parcel_id']].drop_duplicates()

    print(
        f"  📄 Copernicus : "
        f"{len(df_coords)} parcelles"
    )

    # --------------------------------------------------------
    # Charger le GeoJSON
    # --------------------------------------------------------

    import geopandas as gpd

    try:

        gdf = gpd.read_file(
            "Beja_parcelles_800.geojson"
        )

        coords_dict = {}

        for _, row in gdf.iterrows():

            pid = row['parcel_id']

            bounds = row.geometry.bounds

            minx, miny, maxx, maxy = bounds

            coords_dict[pid] = {

                'lon': (minx + maxx) / 2,

                'lat': (miny + maxy) / 2,

                'lon_min': minx,

                'lat_min': miny,

                'lon_max': maxx,

                'lat_max': maxy
            }

        df_coords = pd.DataFrame.from_dict(
            coords_dict,
            orient='index'
        ).reset_index()

        df_coords.columns = [
            'parcel_id',
            'lon',
            'lat',
            'lon_min',
            'lat_min',
            'lon_max',
            'lat_max'
        ]

        print(
            f"  ✅ Coordonnées GeoJSON : "
            f"{len(df_coords)} parcelles"
        )

    except FileNotFoundError:

        print(
            "  ⚠️ GeoJSON introuvable."
        )

        print(
            "  ⚠️ Utilisation de coordonnées "
            "approximatives."
        )

        df_coords['lon'] = 9.1856
        df_coords['lat'] = 36.7256

        df_coords['lon_min'] = (
            df_coords['lon'] - 0.002
        )

        df_coords['lon_max'] = (
            df_coords['lon'] + 0.002
        )

        df_coords['lat_min'] = (
            df_coords['lat'] - 0.002
        )

        df_coords['lat_max'] = (
            df_coords['lat'] + 0.002
        )

except Exception as e:

    print(
        f"  ❌ Erreur coordonnées : {e}"
    )

    exit(1)


# ============================================================
# 4. FUSION
# ============================================================

print("\n🔗 Fusion des données...")

df = df_comp.merge(
    df_gee,
    on='parcel_id',
    how='inner'
)

df = df.merge(
    df_coords,
    on='parcel_id',
    how='inner'
)

print(
    f"  ✅ {len(df)} parcelles après fusion"
)

if len(df) == 0:

    print(
        "\n❌ Aucune donnée après fusion."
    )

    print(
        "Vérifiez que les parcel_id "
        "correspondent."
    )

    exit(1)


# ============================================================
# 5. NETTOYAGE APRÈS FUSION
# ============================================================

# Sécurité supplémentaire :
# convertir à nouveau les booléens après fusion.

df['accord'] = df['accord'].apply(
    convertir_bool
)

df['gee_anomalie'] = df['gee_anomalie'].apply(
    convertir_bool
)

df['api_anomalie'] = df['api_anomalie'].apply(
    convertir_bool
)


# ============================================================
# 6. CALCUL DES INFORMATIONS DE CONCORDANCE
# ============================================================

def calculer_concordance(row):

    """
    Détermine la couleur et le label
    selon la concordance réelle.
    """

    accord = convertir_bool(row['accord'])

    gee_anomalie = convertir_bool(
        row['gee_anomalie']
    )

    api_anomalie = convertir_bool(
        row['api_anomalie']
    )

    # --------------------------------------------------------
    # ACCORD
    # --------------------------------------------------------

    if accord:

        if gee_anomalie or api_anomalie:

            return {
                'couleur': '#4CAF50',
                'label': 'Accord - Anomalie détectée',
                'confiance': 'Très élevée'
            }

        else:

            return {
                'couleur': '#8BC34A',
                'label': 'Accord - Normale',
                'confiance': 'Très élevée'
            }

    # --------------------------------------------------------
    # DIVERGENCE
    # --------------------------------------------------------

    gee_z = valeur_float(
        row['gee_z_score']
    )

    api_z = valeur_float(
        row['api_z_score']
    )

    diff_z = abs(gee_z - api_z)

    if diff_z < 1.0:

        return {
            'couleur': '#FFC107',
            'label': 'Divergence légère (Z < 1)',
            'confiance': 'Modérée'
        }

    else:

        return {
            'couleur': '#F44336',
            'label': 'Divergence forte (Z > 1)',
            'confiance': 'Faible'
        }


print("\n⚙️ Calcul des concordances...")

df['concordance_info'] = df.apply(
    calculer_concordance,
    axis=1
)


# ============================================================
# 7. GÉNÉRATION DU JSON
# ============================================================

print("\n📝 Génération du JSON...")

resultats = []

for _, row in df.iterrows():

    conc = row['concordance_info']

    gee_anomalie = convertir_bool(
        row['gee_anomalie']
    )

    api_anomalie = convertir_bool(
        row['api_anomalie']
    )

    accord = convertir_bool(
        row['accord']
    )

    gee_z_score = valeur_float(
        row['gee_z_score']
    )

    api_z_score = valeur_float(
        row['api_z_score']
    )

    resultats.append({

        # ----------------------------------------------------
        # IDENTIFICATION
        # ----------------------------------------------------

        'parcel_id': str(row['parcel_id']),

        # ----------------------------------------------------
        # COORDONNÉES
        # ----------------------------------------------------

        'lon': valeur_float(row['lon']),

        'lat': valeur_float(row['lat']),

        'lon_min': valeur_float(row['lon_min']),

        'lat_min': valeur_float(row['lat_min']),

        'lon_max': valeur_float(row['lon_max']),

        'lat_max': valeur_float(row['lat_max']),

        # ----------------------------------------------------
        # GEE
        # ----------------------------------------------------

        'gee': {

            'anomalie': gee_anomalie,

            'z_score': gee_z_score,

            'type_anomalie': valeur_str(
                row['gee_type']
            ),

            'criticite': valeur_str(
                row['gee_criticite']
            ),

            'criticite_pct': valeur_float(
                row['gee_criticite_pct']
            )
        },

        # ----------------------------------------------------
        # COPERNICUS
        # ----------------------------------------------------

        'copernicus': {

            'anomalie': api_anomalie,

            'z_score': api_z_score,

            'type_anomalie': valeur_str(
                row['api_type']
            ),

            'criticite': valeur_str(
                row.get(
                    'api_criticite',
                    'Faible'
                )
            )
        },

        # ----------------------------------------------------
        # COMPARAISON
        # ----------------------------------------------------

        'comparaison': {

            'accord': accord,

            'couleur': conc['couleur'],

            'label': conc['label'],

            'confiance': conc['confiance'],

            'difference_z_score': abs(
                gee_z_score - api_z_score
            )
        }
    })


# ============================================================
# 8. CRÉATION DU DOSSIER DE SORTIE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. SAUVEGARDE
# ============================================================

with open(
    OUTPUT_FILE,
    'w',
    encoding='utf-8'
) as f:

    json.dump(
        resultats,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"\n✅ Fichier généré : "
    f"{OUTPUT_FILE}"
)

print(
    f"   {len(resultats)} parcelles exportées"
)


# ============================================================
# 10. STATISTIQUES
# ============================================================

nb_total = len(resultats)

nb_accord = sum(
    1
    for r in resultats
    if r['comparaison']['accord'] is True
)

nb_divergence = sum(
    1
    for r in resultats
    if r['comparaison']['accord'] is False
)

taux_accord = (
    100 * nb_accord / nb_total
    if nb_total > 0
    else 0
)

taux_divergence = (
    100 * nb_divergence / nb_total
    if nb_total > 0
    else 0
)

nb_anomalies_gee = sum(
    1
    for r in resultats
    if r['gee']['anomalie'] is True
)

nb_anomalies_cop = sum(
    1
    for r in resultats
    if r['copernicus']['anomalie'] is True
)


# ============================================================
# 11. STATISTIQUES FINALES
# ============================================================

print("\n" + "=" * 60)
print("📊 STATISTIQUES")
print("=" * 60)

print(
    f"Total parcelles       : "
    f"{nb_total}"
)

print(
    f"Accord                : "
    f"{nb_accord} ({taux_accord:.1f}%)"
)

print(
    f"Divergences           : "
    f"{nb_divergence} ({taux_divergence:.1f}%)"
)

print(
    f"Anomalies GEE         : "
    f"{nb_anomalies_gee}"
)

print(
    f"Anomalies Copernicus  : "
    f"{nb_anomalies_cop}"
)


# ============================================================
# 12. CONTRÔLE FINAL
# ============================================================

print("\n🔎 CONTRÔLE FINAL")

# Comparaison avec le taux calculé directement
# depuis le CSV comparaison.

difference_taux = abs(
    taux_accord - taux_csv
)

print(
    f"  Taux CSV comparaison : "
    f"{taux_csv:.1f}%"
)

print(
    f"  Taux JSON généré     : "
    f"{taux_accord:.1f}%"
)

print(
    f"  Différence            : "
    f"{difference_taux:.2f}%"
)

if difference_taux < 0.01:

    print(
        "  ✅ Les deux taux correspondent."
    )

else:

    print(
        "  ⚠️ ATTENTION : les taux ne correspondent pas !"
    )

    print(
        "  Vérifiez le fichier comparaison."
    )


# ============================================================
# 13. AFFICHER QUELQUES DIVERGENCES
# ============================================================

divergences = [
    r
    for r in resultats
    if r['comparaison']['accord'] is False
]

print(
    f"\n🔴 Exemples de divergences : "
    f"{len(divergences)} trouvées"
)

for r in divergences[:10]:

    print(
        f"  {r['parcel_id']} : "
        f"GEE={r['gee']['anomalie']} | "
        f"Copernicus={r['copernicus']['anomalie']} | "
        f"ΔZ={r['comparaison']['difference_z_score']:.3f}"
    )


print("\n" + "=" * 60)
print("✅ PRÊT POUR LE FRONTEND")
print("=" * 60)

print("\nProchaine étape :")

print(
    "  1. Lancez : "
    "python -m http.server 8080"
)

print(
    "  2. Ouvrez : "
    "http://localhost:8080"
)

print(
    "  3. Rechargez complètement la page "
    "(Ctrl + F5)"
)

