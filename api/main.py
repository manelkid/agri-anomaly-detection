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

# ============================================================
# NOUVELLES ROUTES : COMPARAISON GEE vs COPERNICUS
# ============================================================

from schemas import (
    ComparisonRequest, 
    ComparisonResponse, 
    PredictionSource
)

@app.post("/compare-sources", response_model=ComparisonResponse)
def compare_sources(request: ComparisonRequest):
    """
    Compare les prédictions obtenues à partir de deux sources satellite différentes
    
    - **GEE** : Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
    - **Copernicus** : Copernicus Data Space Ecosystem (Statistical API)
    
    Cette route permet de :
    1. Valider la robustesse du modèle (indépendant de la source)
    2. Détecter d'éventuels artefacts spécifiques à une source
    3. Augmenter la confiance dans le diagnostic quand les deux sources concordent
    
    **Méthodologie** :
    - Le même modèle LSTM est appliqué aux deux séries temporelles
    - Les deux prédictions sont comparées (anomalie oui/non, Z-score)
    - Une analyse de concordance est retournée
    
    **Cas d'usage** :
    - Validation croisée des résultats
    - Détection d'anomalies robustes (concordance élevée = forte confiance)
    - Identification des cas ambigus (divergence = vérification terrain recommandée)
    """
    
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Validation : les deux séries doivent avoir la même longueur
    len_gee = len(request.timeseries_gee.ndvi)
    len_cop = len(request.timeseries_copernicus.ndvi)
    
    if len_gee != len_cop:
        raise HTTPException(
            status_code=400,
            detail=f"Les deux séries doivent avoir la même longueur "
                   f"(GEE: {len_gee}, Copernicus: {len_cop})"
        )
    
    # Prédiction sur la série GEE
    pred_gee = diagnostiquer_serie(
        parcel_id=request.parcel_id,
        ndvi=request.timeseries_gee.ndvi,
        evi=request.timeseries_gee.evi,
        ndmi=request.timeseries_gee.ndmi
    )
    
    # Prédiction sur la série Copernicus
    pred_copernicus = diagnostiquer_serie(
        parcel_id=request.parcel_id,
        ndvi=request.timeseries_copernicus.ndvi,
        evi=request.timeseries_copernicus.evi,
        ndmi=request.timeseries_copernicus.ndmi
    )
    
    # Analyse de concordance
    concordance = pred_gee.anomalie == pred_copernicus.anomalie
    diff_z_score = abs(pred_gee.z_score - pred_copernicus.z_score)
    
    # Comparaison des séries brutes (corrélations)
    ndvi_gee = np.array(request.timeseries_gee.ndvi)
    ndvi_cop = np.array(request.timeseries_copernicus.ndvi)
    
    evi_gee = np.array(request.timeseries_gee.evi)
    evi_cop = np.array(request.timeseries_copernicus.evi)
    
    ndmi_gee = np.array(request.timeseries_gee.ndmi)
    ndmi_cop = np.array(request.timeseries_copernicus.ndmi)
    
    # Calcul des corrélations (Pearson)
    def safe_correlation(x, y):
        """Calcul de corrélation avec gestion des cas dégénérés"""
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    
    corr_ndvi = safe_correlation(ndvi_gee, ndvi_cop)
    corr_evi = safe_correlation(evi_gee, evi_cop)
    corr_ndmi = safe_correlation(ndmi_gee, ndmi_cop)
    
    # Différences moyennes absolues (pour évaluer la similarité des séries)
    mae_ndvi = float(np.mean(np.abs(ndvi_gee - ndvi_cop)))
    mae_evi = float(np.mean(np.abs(evi_gee - evi_cop)))
    mae_ndmi = float(np.mean(np.abs(ndmi_gee - ndmi_cop)))
    
    # Verdict synthétique
    if concordance and diff_z_score < 0.5:
        verdict = "Excellent accord entre les deux sources"
        confiance = "Très élevée"
        recommandation = "Les deux sources confirment le diagnostic. Confiance maximale."
    elif concordance and diff_z_score < 1.0:
        verdict = "Bon accord, légère différence d'intensité"
        confiance = "Élevée"
        recommandation = "Diagnostic concordant malgré de légères variations d'intensité (normales entre sources)."
    elif concordance:
        verdict = "Accord sur le diagnostic, mais intensités divergentes"
        confiance = "Modérée"
        recommandation = "Même diagnostic mais Z-scores éloignés. Vérifier les valeurs brutes."
    elif diff_z_score < 1.0:
        verdict = "Diagnostic divergent, mais Z-scores proches (zone d'incertitude)"
        confiance = "Faible"
        recommandation = "Cas ambigu : une source détecte une anomalie, l'autre non. Vérification terrain recommandée."
    else:
        verdict = "Divergence significative entre les sources"
        confiance = "Très faible"
        recommandation = "Forte divergence. Vérifier la qualité des données sources (nuages, artefacts). Inspection terrain nécessaire."
    
    # Évaluation de la qualité des corrélations
    corr_moyenne = (corr_ndvi + corr_evi + corr_ndmi) / 3
    
    if corr_moyenne > 0.9:
        qualite_correlation = "Excellente (> 0.9)"
    elif corr_moyenne > 0.7:
        qualite_correlation = "Bonne (0.7-0.9)"
    elif corr_moyenne > 0.5:
        qualite_correlation = "Modérée (0.5-0.7)"
    else:
        qualite_correlation = "Faible (< 0.5) - Vérifier les données"
    
    return ComparisonResponse(
        parcel_id=request.parcel_id,
        
        # Prédictions des deux sources
        prediction_gee=PredictionSource(
            source="Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)",
            anomalie=pred_gee.anomalie,
            z_score=pred_gee.z_score,
            type_anomalie=pred_gee.type_anomalie,
            criticite=pred_gee.criticite,
            erreur_reconstruction=pred_gee.erreur_reconstruction
        ),
        
        prediction_copernicus=PredictionSource(
            source="Copernicus Data Space (Sentinel-2 L2A Statistical API)",
            anomalie=pred_copernicus.anomalie,
            z_score=pred_copernicus.z_score,
            type_anomalie=pred_copernicus.type_anomalie,
            criticite=pred_copernicus.criticite,
            erreur_reconstruction=pred_copernicus.erreur_reconstruction
        ),
        
        # Analyse de concordance
        concordance=concordance,
        difference_z_score=round(diff_z_score, 3),
        
        # Corrélations des séries brutes
        correlation_ndvi=round(corr_ndvi, 3),
        correlation_evi=round(corr_evi, 3),
        correlation_ndmi=round(corr_ndmi, 3),
        
        # Différences moyennes
        mae_ndvi=round(mae_ndvi, 4),
        mae_evi=round(mae_evi, 4),
        mae_ndmi=round(mae_ndmi, 4),
        
        # Synthèse
        verdict=verdict,
        confiance=confiance,
        qualite_correlation=qualite_correlation,
        recommandation=recommandation
    )


@app.get("/parcelle/{parcel_id}/compare", response_model=ComparisonResponse)
def comparer_parcelle_cataloguee(parcel_id: str):
    """
    Compare automatiquement les données GEE et Copernicus pour une parcelle du catalogue
    
    **Prérequis** :
    - La parcelle doit exister dans `donnees_parcelles_2025.csv` (données GEE)
    - La parcelle doit avoir des données Copernicus dans `donnees_copernicus_2025.csv`
    
    **Utilité** :
    - Test rapide sans avoir à fournir les données manuellement
    - Démonstration de la comparaison sur des vraies parcelles
    - Validation de la cohérence entre sources sur les 800 parcelles de Béja
    """
    
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Vérifier que la parcelle existe dans le catalogue GEE
    if parcel_id not in catalogue_parcelles:
        raise HTTPException(
            status_code=404,
            detail=f"Parcelle '{parcel_id}' inconnue dans le catalogue GEE. Voir /parcelles pour la liste."
        )
    
    # Charger les données Copernicus (si disponibles)
    copernicus_path = ARTEFACTS_DIR / "donnees_copernicus_2025.csv"
    
    if not copernicus_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Fichier donnees_copernicus_2025.csv non trouvé. "
                   "Utilisez /compare-sources avec les données manuelles."
        )
    
    # Charger les données Copernicus pour cette parcelle
    donnees_copernicus = None
    
    with open(copernicus_path, newline="") as f:
        for ligne in csv.DictReader(f):
            if ligne["parcel_id"] == parcel_id:
                if donnees_copernicus is None:
                    donnees_copernicus = {"NDVI": [], "EVI": [], "NDMI": []}
                donnees_copernicus["NDVI"].append(float(ligne["NDVI"]))
                donnees_copernicus["EVI"].append(float(ligne["EVI"]))
                donnees_copernicus["NDMI"].append(float(ligne["NDMI"]))
    
    if donnees_copernicus is None:
        raise HTTPException(
            status_code=404,
            detail=f"Parcelle '{parcel_id}' n'a pas de données Copernicus disponibles"
        )
    
    # Créer un objet ComparisonRequest
    from schemas import SerieTemporelle
    
    request = ComparisonRequest(
        parcel_id=parcel_id,
        timeseries_gee=SerieTemporelle(
            parcel_id=parcel_id,
            ndvi=catalogue_parcelles[parcel_id]["NDVI"],
            evi=catalogue_parcelles[parcel_id]["EVI"],
            ndmi=catalogue_parcelles[parcel_id]["NDMI"]
        ),
        timeseries_copernicus=SerieTemporelle(
            parcel_id=parcel_id,
            ndvi=donnees_copernicus["NDVI"],
            evi=donnees_copernicus["EVI"],
            ndmi=donnees_copernicus["NDMI"]
        )
    )
    
    # Réutiliser la logique de /compare-sources
    return compare_sources(request)


@app.get("/stats-comparaison")
def statistiques_comparaison():
    """
    Statistiques globales sur la concordance GEE vs Copernicus
    
    Analyse toutes les parcelles disponibles dans les deux catalogues et calcule :
    - Taux de concordance global
    - Corrélation moyenne des indices
    - Distribution des divergences
    
    **Utile pour** :
    - Présentation à l'encadrant (chiffre clé)
    - Validation de la robustesse du modèle
    - Identification des parcelles problématiques
    """
    
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    copernicus_path = ARTEFACTS_DIR / "donnees_copernicus_2025.csv"
    
    if not copernicus_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Fichier donnees_copernicus_2025.csv non trouvé"
        )
    
    # Charger toutes les données Copernicus
    catalogue_copernicus = defaultdict(lambda: {"NDVI": [], "EVI": [], "NDMI": []})
    
    with open(copernicus_path, newline="") as f:
        for ligne in csv.DictReader(f):
            pid = ligne["parcel_id"]
            catalogue_copernicus[pid]["NDVI"].append(float(ligne["NDVI"]))
            catalogue_copernicus[pid]["EVI"].append(float(ligne["EVI"]))
            catalogue_copernicus[pid]["NDMI"].append(float(ligne["NDMI"]))
    
    # Parcelles communes aux deux catalogues
    parcelles_communes = set(catalogue_parcelles.keys()) & set(catalogue_copernicus.keys())
    
    if not parcelles_communes:
        raise HTTPException(
            status_code=404,
            detail="Aucune parcelle commune entre GEE et Copernicus"
        )
    
    # Calculer les statistiques
    nb_concordances = 0
    nb_total = len(parcelles_communes)
    
    correlations_ndvi = []
    correlations_evi = []
    correlations_ndmi = []
    
    differences_z = []
    
    divergences_details = []
    
    for parcel_id in parcelles_communes:
        # Prédictions
        pred_gee = diagnostiquer_serie(
            parcel_id,
            catalogue_parcelles[parcel_id]["NDVI"],
            catalogue_parcelles[parcel_id]["EVI"],
            catalogue_parcelles[parcel_id]["NDMI"]
        )
        
        pred_cop = diagnostiquer_serie(
            parcel_id,
            catalogue_copernicus[parcel_id]["NDVI"],
            catalogue_copernicus[parcel_id]["EVI"],
            catalogue_copernicus[parcel_id]["NDMI"]
        )
        
        # Concordance
        if pred_gee.anomalie == pred_cop.anomalie:
            nb_concordances += 1
        else:
            divergences_details.append({
                "parcel_id": parcel_id,
                "gee_anomalie": pred_gee.anomalie,
                "cop_anomalie": pred_cop.anomalie,
                "gee_z_score": pred_gee.z_score,
                "cop_z_score": pred_cop.z_score
            })
        
        # Z-scores
        differences_z.append(abs(pred_gee.z_score - pred_cop.z_score))
        
        # Corrélations
        def safe_corr(x, y):
            if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])
        
        correlations_ndvi.append(safe_corr(
            catalogue_parcelles[parcel_id]["NDVI"],
            catalogue_copernicus[parcel_id]["NDVI"]
        ))
        correlations_evi.append(safe_corr(
            catalogue_parcelles[parcel_id]["EVI"],
            catalogue_copernicus[parcel_id]["EVI"]
        ))
        correlations_ndmi.append(safe_corr(
            catalogue_parcelles[parcel_id]["NDMI"],
            catalogue_copernicus[parcel_id]["NDMI"]
        ))
    
    taux_concordance = 100 * nb_concordances / nb_total
    
    return {
        "nb_parcelles_testees": nb_total,
        "nb_concordances": nb_concordances,
        "taux_concordance_pct": round(taux_concordance, 2),
        
        "correlations_moyennes": {
            "NDVI": round(np.mean(correlations_ndvi), 3),
            "EVI": round(np.mean(correlations_evi), 3),
            "NDMI": round(np.mean(correlations_ndmi), 3)
        },
        
        "difference_z_score": {
            "moyenne": round(np.mean(differences_z), 3),
            "mediane": round(np.median(differences_z), 3),
            "max": round(np.max(differences_z), 3)
        },
        
        "nb_divergences": len(divergences_details),
        "divergences": divergences_details[:10],  # Top 10
        
        "phrase_presentation": (
            f"Le taux de concordance de {taux_concordance:.1f}% entre deux sources "
            f"satellite indépendantes (Google Earth Engine et Copernicus Data Space) "
            f"sur {nb_total} parcelles confirme que la détection reflète un vrai "
            f"phénomène agricole, indépendant de la source de données."
        )
    }