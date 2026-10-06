"""
Compare le diagnostic d'anomalie pour les MÊMES parcelles, selon que les
données viennent de Google Earth Engine ou de Copernicus Data Space.

Prérequis :
  - L'API tourne en local (python -m uvicorn main:app --port 8000)
  - copernicus_donnees.csv généré (via copernicus_extraction.py)
  - donnees_parcelles_2025.csv disponible (export GEE existant, celui
    utilisé par l'API pour /parcelle/{id} — copie-le à côté de ce script)

Usage : python comparer_sources.py
"""

import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"
GEE_CSV = "data/donnees_parcelles_2025.csv"
COPERNICUS_CSV = "data/copernicus_donnees_completes.csv"


def construire_payload(df: pd.DataFrame, parcel_id: str) -> dict | None:
    sous_ensemble = df[df["parcel_id"] == parcel_id].sort_values("date")
    if len(sous_ensemble) != 12:
        print(f"  {parcel_id} : {len(sous_ensemble)}/12 points -> ignorée")
        return None
    return {
        "parcel_id": parcel_id,
        "ndvi": sous_ensemble["NDVI"].tolist(),
        "evi": sous_ensemble["EVI"].tolist(),
        "ndmi": sous_ensemble["NDMI"].tolist(),
    }


def diagnostiquer(payload: dict) -> dict:
    reponse = requests.post(API_URL, json=payload)
    reponse.raise_for_status()
    return reponse.json()


def main():
    df_gee = pd.read_csv(GEE_CSV)
    df_cop = pd.read_csv(COPERNICUS_CSV)

    parcelles_communes = sorted(set(df_gee["parcel_id"]) & set(df_cop["parcel_id"]))
    print(f"Parcelles présentes dans les deux sources : {parcelles_communes}\n")

    resultats = []
    for parcel_id in parcelles_communes:
        payload_gee = construire_payload(df_gee, parcel_id)
        payload_cop = construire_payload(df_cop, parcel_id)
        if payload_gee is None or payload_cop is None:
            continue

        diag_gee = diagnostiquer(payload_gee)
        diag_cop = diagnostiquer(payload_cop)

        accord = diag_gee["anomalie"] == diag_cop["anomalie"]

        resultats.append({
            "parcel_id": parcel_id,
            "anomalie_GEE": diag_gee["anomalie"],
            "cause_GEE": diag_gee["type_anomalie"],
            "z_GEE": diag_gee["z_score"],
            "anomalie_Copernicus": diag_cop["anomalie"],
            "cause_Copernicus": diag_cop["type_anomalie"],
            "z_Copernicus": diag_cop["z_score"],
            "ACCORD": "OUI" if accord else "NON",
        })

    df_resultats = pd.DataFrame(resultats)
    print(df_resultats.to_string(index=False))

    taux_accord = (df_resultats["ACCORD"] == "OUI").mean() * 100
    print(f"\nTaux d'accord entre les deux sources : {taux_accord:.1f}%")

    df_resultats.to_csv("data/comparaison_gee_copernicus.csv", index=False)
    print("Résultats sauvegardés : comparaison_gee_copernicus.csv")


if __name__ == "__main__":
    main()