"""
API de détection d'anomalies agricoles.

Endpoints :
  GET  /health   -> vérifie que le modèle est chargé
  POST /predict  -> reçoit une série NDVI/EVI/NDMI, retourne le diagnostic

Lancement local :
  uvicorn main:app --reload --host 0.0.0.0 --port 7860
"""

import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from model import LSTMAutoencoder
from schemas import SerieTemporelle, ResultatPrediction

app = FastAPI(
    title="Détection d'anomalies agricoles - Béja",
    description="API servant le modèle LSTM Autoencodeur entraîné sur Sentinel-2 (NDVI/EVI/NDMI).",
    version="1.0.0",
)

ARTEFACTS_DIR = Path(__file__).parent / "artefacts"

# --- Chargement des artefacts au démarrage (une seule fois) ---
modele = None
seuils = None


@app.on_event("startup")
def charger_artefacts():
    global modele, seuils

    with open(ARTEFACTS_DIR / "seuils.json") as f:
        seuils = json.load(f)

    modele = LSTMAutoencoder(
        n_features=len(seuils["indicateurs"]),
        hidden=seuils["hidden_dim"],
        latent=seuils["latent_dim"],
    )
    modele.load_state_dict(torch.load(ARTEFACTS_DIR / "model.pt", map_location="cpu"))
    modele.eval()

    print(f"Modèle chargé : {seuils['indicateurs']}, hidden={seuils['hidden_dim']}")


@app.get("/health")
def health():
    return {
        "status": "ok" if modele is not None else "modele_non_charge",
        "indicateurs": seuils["indicateurs"] if seuils else None,
    }


def classer_criticite(z: float) -> str:
    if z < 2:
        return "Faible"
    elif z < 3:
        return "Modéré"
    elif z < 4:
        return "Élevé"
    return "Critique"


@app.post("/predict", response_model=ResultatPrediction)
def predict(serie: SerieTemporelle):
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    longueurs = {len(serie.ndvi), len(serie.evi), len(serie.ndmi)}
    if len(longueurs) != 1:
        raise HTTPException(status_code=400, detail="ndvi, evi et ndmi doivent avoir la même longueur")

    # --- Construire le tenseur (1, n_dates, 3) dans le même ordre que l'entraînement ---
    # Normalisation : chaque indice est ramené à une référence GLOBALE stable
    # (moyenne/écart-type calculés une fois pour toutes sur l'ensemble des
    # parcelles d'entraînement), jamais recalculée à partir des données à
    # tester elles-mêmes — ça évite qu'une vraie anomalie ne dilue sa propre
    # détection en tirant la moyenne de référence vers elle.
    indicateurs = seuils["indicateurs"]  # ex: ['NDVI', 'EVI', 'NDMI']
    donnees_brutes = {"NDVI": serie.ndvi, "EVI": serie.evi, "NDMI": serie.ndmi}

    colonnes = []
    for ind in indicateurs:
        valeurs = np.array(donnees_brutes[ind], dtype=float)
        ref = seuils["stats_globales"][ind]
        ecart = ref["ecart_type"] if ref["ecart_type"] > 0 else 1e-6
        valeurs_norm = (valeurs - ref["moyenne"]) / ecart
        colonnes.append(valeurs_norm)

    x = np.stack(colonnes, axis=-1)  # (n_dates, n_features)
    x_tensor = torch.FloatTensor(x).unsqueeze(0)  # (1, n_dates, n_features)

    # --- Reconstruction + erreur ---
    with torch.no_grad():
        reconstruction = modele(x_tensor)
        erreur = torch.mean((x_tensor - reconstruction) ** 2).item()

    # --- Z-score par rapport à la distribution observée à l'entraînement ---
    z_score = (erreur - seuils["mean_test"]) / seuils["std_test"] if seuils["std_test"] > 0 else 0.0
    est_anomalie = z_score > seuils["z_seuil"]
    criticite_pct = round(100 * min(max(z_score, 0), 5) / 5, 1)

    return ResultatPrediction(
        parcel_id=serie.parcel_id,
        erreur_reconstruction=round(erreur, 6),
        z_score=round(z_score, 3),
        anomalie=bool(est_anomalie),
        criticite=classer_criticite(z_score),
        criticite_pct=criticite_pct,
    )
