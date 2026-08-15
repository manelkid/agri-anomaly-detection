"""
Envoie les données récentes (12 derniers mois) d'une ou plusieurs parcelles
à l'API locale, et affiche le diagnostic retourné.

Prérequis :
  - L'API doit tourner en local (python -m uvicorn main:app --port 7860)
  - Le fichier beja_recent_12mois.csv doit être téléchargé depuis Drive
    et placé à côté de ce script (ou ajuster CSV_PATH ci-dessous)

Usage : python demo_predict_recent.py
"""

import pandas as pd
import requests

CSV_PATH = "beja_recent_12mois.csv"
API_URL = "http://localhost:8000/predict"


def construire_payload(df: pd.DataFrame, parcel_id: str) -> dict:
    parcelle = df[df["parcel_id"] == parcel_id].sort_values("date")
    if len(parcelle) != 12:
        raise ValueError(
            f"{parcel_id} a {len(parcelle)} points (12 attendus). "
            f"Vérifie que l'export GEE a bien produit 12 mois complets."
        )
    return {
        "parcel_id": parcel_id,
        "ndvi": parcelle["NDVI"].tolist(),
        "evi": parcelle["EVI"].tolist(),
        "ndmi": parcelle["NDMI"].tolist(),
    }


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Données chargées : {len(df)} lignes, {df['parcel_id'].nunique()} parcelles")

    parcelles_disponibles = sorted(df["parcel_id"].unique())
    print(f"Parcelles disponibles : {parcelles_disponibles}\n")

    for parcel_id in parcelles_disponibles:
        try:
            payload = construire_payload(df, parcel_id)
        except ValueError as e:
            print(f"{parcel_id} : ignorée ({e})")
            continue

        reponse = requests.post(API_URL, json=payload)

        if reponse.status_code == 200:
            resultat = reponse.json()
            statut = "ANOMALIE" if resultat["anomalie"] else "normal"
            print(f"{parcel_id} -> {statut} "
                  f"(z={resultat['z_score']}, criticité={resultat['criticite']})")
        else:
            print(f"{parcel_id} -> erreur API {reponse.status_code} : {reponse.text}")


if __name__ == "__main__":
    main()
