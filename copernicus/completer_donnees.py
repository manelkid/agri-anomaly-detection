"""
Complète les séries Copernicus à exactement 12 mois par parcelle
(certains mois peuvent manquer si aucune image n'était disponible
sous le seuil de nuages, ex: décembre trop nuageux).

Usage : python completer_donnees.py
"""

import pandas as pd

FICHIER_ENTREE = "data/copernicus_donnees.csv"
FICHIER_SORTIE = "data/copernicus_donnees_completes.csv"

# Les 12 mois attendus (doit correspondre exactement à la période
# utilisée pour donnees_parcelles_2025.csv, côté GEE)
MOIS_ATTENDUS = pd.date_range("2025-01-01", "2025-12-01", freq="MS")


def main():
    df = pd.read_csv(FICHIER_ENTREE)
    df["date"] = pd.to_datetime(df["date"])

    lignes_completes = []
    parcelles_incompletes = []

    for parcel_id, groupe in df.groupby("parcel_id"):
        groupe = groupe.set_index("date").reindex(MOIS_ATTENDUS)
        nb_manquants = groupe["NDVI"].isna().sum()

        if nb_manquants > 0:
            parcelles_incompletes.append((parcel_id, nb_manquants))

        # Interpolation linéaire + comblement des bords (si le mois
        # manquant est au tout début ou à la toute fin de la série)
        groupe[["NDVI", "EVI", "NDMI"]] = (
            groupe[["NDVI", "EVI", "NDMI"]]
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )
        groupe["parcel_id"] = parcel_id
        groupe = groupe.reset_index().rename(columns={"index": "date"})
        lignes_completes.append(groupe)

    df_final = pd.concat(lignes_completes, ignore_index=True)
    df_final["date"] = df_final["date"].dt.strftime("%Y-%m-%d")
    df_final = df_final[["parcel_id", "date", "NDVI", "EVI", "NDMI"]]

    df_final.to_csv(FICHIER_SORTIE, index=False)

    print(f"Parcelles avec des mois comblés par interpolation : {len(parcelles_incompletes)}")
    for pid, n in parcelles_incompletes:
        print(f"  {pid} : {n} mois comblé(s)")

    print(f"\n{df_final['parcel_id'].nunique()} parcelles x 12 mois -> {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()