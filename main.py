"""
API de détection d'anomalies agricoles.

Endpoints :
  GET  /health   -> vérifie que le modèle est chargé
  POST /predict  -> reçoit une série NDVI/EVI/NDMI, retourne le diagnostic

Lancement local :
  uvicorn main:app --reload --host 0.0.0.0 --port 7860
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from model import LSTMAutoencoder
from schemas import SerieTemporelle, ResultatPrediction, InfoModele

app = FastAPI(
    title="Détection d'anomalies agricoles - Béja",
    description="API servant le modèle LSTM Autoencodeur entraîné sur Sentinel-2 (NDVI/EVI/NDMI).",
    version="1.0.0",
)

ARTEFACTS_DIR = Path(__file__).parent / "artefacts"

# --- Chargement des artefacts au démarrage (une seule fois) ---
modele = None
seuils = None
catalogue_parcelles = {}  # {parcel_id: {"NDVI": [...], "EVI": [...], "NDMI": [...]}}


@app.on_event("startup")
def charger_artefacts():
    global modele, seuils, catalogue_parcelles

    with open(ARTEFACTS_DIR / "seuils.json") as f:
        seuils = json.load(f)

    modele = LSTMAutoencoder(
        n_features=len(seuils["indicateurs"]),
        hidden=seuils["hidden_dim"],
        latent=seuils["latent_dim"],
    )
    modele.load_state_dict(torch.load(ARTEFACTS_DIR / "model.pt", map_location="cpu"))
    modele.eval()

    # --- Charger le catalogue des vraies parcelles (données réelles 2025) ---
    donnees_path = ARTEFACTS_DIR / "donnees_parcelles_2025.csv"
    if donnees_path.exists():
        brut = defaultdict(lambda: {"NDVI": [], "EVI": [], "NDMI": []})
        with open(donnees_path, newline="") as f:
            for ligne in csv.DictReader(f):
                pid = ligne["parcel_id"]
                brut[pid]["NDVI"].append(float(ligne["NDVI"]))
                brut[pid]["EVI"].append(float(ligne["EVI"]))
                brut[pid]["NDMI"].append(float(ligne["NDMI"]))
        catalogue_parcelles = dict(brut)
        print(f"Catalogue chargé : {len(catalogue_parcelles)} vraies parcelles disponibles")
    else:
        print("Aucun catalogue de parcelles trouvé (donnees_parcelles_2025.csv absent) — "
              "seule la route /predict manuelle sera disponible.")

    print(f"Modèle chargé : {seuils['indicateurs']}, hidden={seuils['hidden_dim']}")


@app.get("/health")
def health():
    return {
        "status": "ok" if modele is not None else "modele_non_charge",
        "indicateurs": seuils["indicateurs"] if seuils else None,
    }


@app.get("/model-info", response_model=InfoModele)
def model_info():
    """
    Transparence sur le modèle : sur quelles données il a été entraîné et
    validé, pour que l'utilisateur juge lui-même la fiabilité d'un résultat
    selon la période qu'il teste (le modèle n'est PAS automatiquement à
    jour pour des années non vues, ex: 2024+ — voir avertissement).
    """
    if seuils is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return InfoModele(
        indicateurs=seuils["indicateurs"],
        periode_entrainement="2018-2023 (6 ans, région de Béja, Tunisie)",
        periode_validation="2024-2025",
        hidden_dim=seuils["hidden_dim"],
        latent_dim=seuils["latent_dim"],
        z_seuil=seuils["z_seuil"],
        avertissement=(
            "Ce modèle a été entraîné et validé sur des données jusqu'à fin 2025. "
            "Pour des données d'années plus récentes, les résultats restent "
            "calculables mais leur fiabilité n'est pas garantie sans réentraînement "
            "périodique du modèle sur des données plus récentes (dérive du modèle)."
        ),
    )


def classer_criticite(z: float) -> str:
    if z < 2:
        return "Faible"
    elif z < 3:
        return "Modéré"
    elif z < 4:
        return "Élevé"
    return "Critique"


def determiner_cause(ndvi: list, evi: list, ndmi: list, indicateurs: list, colonnes: list, est_anomalie: bool) -> str:
    """
    Version simplifiée de la classification de cause utilisée dans l'analyse
    Colab (sans ajustement saisonnier ni historique personnel, non disponibles
    ici). Donne un mot-clé simple et compréhensible plutôt qu'un score brut.
    """
    if not est_anomalie:
        return "Normal"

    # Donnée improbable : un NDVI négatif n'est presque jamais un vrai signal agricole
    if min(ndvi) < 0:
        return "Donnée suspecte"

    # Écart moyen (normalisé) par indice sur toute la série fournie
    z_par_indice = {ind: float(np.mean(col)) for ind, col in zip(indicateurs, colonnes)}
    z_vigueur = min(z_par_indice.get("NDVI", 0), z_par_indice.get("EVI", 0))
    z_eau = z_par_indice.get("NDMI", 0)
    SEUIL = 0.75

    if z_vigueur < -SEUIL and z_eau < -SEUIL:
        return "Sécheresse"
    elif z_vigueur < -SEUIL and z_eau >= -0.4:
        return "Maladie ou dégât"
    elif z_eau < -SEUIL and z_vigueur >= -0.4:
        return "Stress hydrique léger"
    elif z_vigueur > SEUIL and z_eau > SEUIL:
        return "Vigueur anormalement élevée"
    else:
        return "Anomalie légère"


def diagnostiquer_serie(parcel_id: str, ndvi: list, evi: list, ndmi: list) -> ResultatPrediction:
    """Logique de prédiction centrale, réutilisée par /predict et /parcelle/{id}."""
    longueurs = {len(ndvi), len(evi), len(ndmi)}
    if len(longueurs) != 1:
        raise HTTPException(status_code=400, detail="ndvi, evi et ndmi doivent avoir la même longueur")

    # Normalisation : référence GLOBALE stable (calculée une fois sur le train),
    # jamais recalculée à partir des données testées (évite qu'une anomalie
    # ne dilue sa propre détection).
    indicateurs = seuils["indicateurs"]
    donnees_brutes = {"NDVI": ndvi, "EVI": evi, "NDMI": ndmi}

    colonnes = []
    for ind in indicateurs:
        valeurs = np.array(donnees_brutes[ind], dtype=float)
        ref = seuils["stats_globales"][ind]
        ecart = ref["ecart_type"] if ref["ecart_type"] > 0 else 1e-6
        colonnes.append((valeurs - ref["moyenne"]) / ecart)

    x = np.stack(colonnes, axis=-1)
    x_tensor = torch.FloatTensor(x).unsqueeze(0)

    with torch.no_grad():
        reconstruction = modele(x_tensor)
        erreur = torch.mean((x_tensor - reconstruction) ** 2).item()

    z_score = (erreur - seuils["mean_test"]) / seuils["std_test"] if seuils["std_test"] > 0 else 0.0
    est_anomalie = z_score > seuils["z_seuil"]
    cause = determiner_cause(ndvi, evi, ndmi, indicateurs, colonnes, est_anomalie)

    return ResultatPrediction(
        parcel_id=parcel_id,
        erreur_reconstruction=round(erreur, 6),
        z_score=round(z_score, 3),
        anomalie=bool(est_anomalie),
        type_anomalie=cause,
        criticite=classer_criticite(z_score),
        criticite_pct=round(100 * min(max(z_score, 0), 5) / 5, 1),
    )


@app.post("/predict", response_model=ResultatPrediction)
def predict(serie: SerieTemporelle):
    """Prédiction sur une série fournie manuellement (utile pour tester avec
    de nouvelles données, pas forcément une des 800 parcelles connues)."""
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return diagnostiquer_serie(serie.parcel_id, serie.ndvi, serie.evi, serie.ndmi)


@app.get("/parcelles")
def lister_parcelles():
    """Liste les vraies parcelles disponibles (données réelles Béja 2023)."""
    if not catalogue_parcelles:
        raise HTTPException(status_code=404, detail="Aucune donnée de parcelle chargée")
    return {"nb_parcelles": len(catalogue_parcelles), "parcel_ids": sorted(catalogue_parcelles.keys())}


@app.get("/parcelle/{parcel_id}", response_model=ResultatPrediction)
def predire_vraie_parcelle(parcel_id: str):
    """
    Prédiction sur une VRAIE parcelle de Béja (données Sentinel-2 réelles,
    2023) — pas un exemple inventé. C'est la route à utiliser pour une
    démonstration concrète : GET /parcelle/P0703 par exemple.
    """
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    if parcel_id not in catalogue_parcelles:
        raise HTTPException(status_code=404,
                             detail=f"Parcelle '{parcel_id}' inconnue. Voir /parcelles pour la liste.")

    donnees = catalogue_parcelles[parcel_id]
    return diagnostiquer_serie(parcel_id, donnees["NDVI"], donnees["EVI"], donnees["NDMI"])
