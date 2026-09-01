from pydantic import BaseModel, Field, field_validator
from typing import List


class SerieTemporelle(BaseModel):
    """
    Une série temporelle NDVI/EVI/NDMI pour UNE parcelle.
    Doit contenir exactement 12 points (12 mois), valeurs entre -1 et 1
    (plage physique valide pour ces indices).
    """
    parcel_id: str = Field(..., example="P0001")
    ndvi: List[float] = Field(..., example=[0.45, 0.42, 0.38, 0.30, 0.25,
                                             0.22, 0.20, 0.24, 0.30, 0.38, 0.44, 0.46])
    evi: List[float] = Field(..., example=[0.30, 0.28, 0.25, 0.20, 0.17,
                                            0.15, 0.14, 0.16, 0.20, 0.25, 0.29, 0.31])
    ndmi: List[float] = Field(..., example=[0.10, 0.08, 0.05, -0.02, -0.08,
                                             -0.12, -0.14, -0.10, -0.03, 0.04, 0.09, 0.11])

    @field_validator('ndvi', 'evi', 'ndmi')
    @classmethod
    def verifier_serie(cls, v: List[float]) -> List[float]:
        if len(v) != 12:
            raise ValueError(f"La série doit contenir exactement 12 points (1 par mois), reçu : {len(v)}")
        for val in v:
            if not (-1.0 <= val <= 1.0):
                raise ValueError(f"Valeur hors plage physique valide [-1, 1] : {val}")
        return v


class ResultatPrediction(BaseModel):
    parcel_id: str
    erreur_reconstruction: float
    z_score: float
    anomalie: bool
    type_anomalie: str
    criticite: str
    criticite_pct: float


class InfoModele(BaseModel):
    """
    Métadonnées du modèle — indispensable pour qu'un utilisateur de l'API
    sache si un résultat est fiable pour la période qu'il teste.
    """
    indicateurs: List[str]
    periode_entrainement: str
    periode_validation: str
    hidden_dim: int
    latent_dim: int
    z_seuil: float
    avertissement: str

# ============================================================
# SCHÉMAS POUR LA COMPARAISON GEE vs COPERNICUS
# ============================================================

class PredictionSource(BaseModel):
    """
    Prédiction avec identification de la source satellite.
    
    Utilisé dans les réponses de comparaison pour distinguer clairement
    les résultats obtenus à partir de Google Earth Engine vs Copernicus.
    """
    source: str = Field(..., example="Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)")
    anomalie: bool = Field(..., example=False)
    z_score: float = Field(..., example=1.234)
    type_anomalie: str = Field(..., example="Normal")
    criticite: str = Field(..., example="Faible")
    erreur_reconstruction: float = Field(..., example=0.0156)


class ComparisonRequest(BaseModel):
    """
    Requête de comparaison entre deux sources satellite indépendantes.
    
    Permet de valider la robustesse du modèle en comparant les prédictions
    obtenues à partir de :
    - Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
    - Copernicus Data Space Ecosystem (Statistical API)
    
    Les deux séries temporelles doivent avoir la même longueur (idéalement
    12 mois pour une année complète). Cette route est utile pour :
    - Détecter d'éventuels artefacts spécifiques à une source
    - Augmenter la confiance dans le diagnostic (concordance = forte fiabilité)
    - Identifier les cas ambigus nécessitant une vérification terrain
    """
    parcel_id: str = Field(..., example="P0042")
    timeseries_gee: SerieTemporelle
    timeseries_copernicus: SerieTemporelle
    
    class Config:
        schema_extra = {
            "example": {
                "parcel_id": "P0042",
                "timeseries_gee": {
                    "parcel_id": "P0042",
                    "ndvi": [0.45, 0.48, 0.52, 0.58, 0.62, 0.65, 0.61, 0.55, 0.48, 0.42, 0.38, 0.35],
                    "evi": [0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.44, 0.39, 0.34, 0.30, 0.27, 0.25],
                    "ndmi": [0.21, 0.24, 0.27, 0.31, 0.34, 0.37, 0.33, 0.28, 0.23, 0.19, 0.16, 0.14]
                },
                "timeseries_copernicus": {
                    "parcel_id": "P0042",
                    "ndvi": [0.46, 0.49, 0.53, 0.59, 0.63, 0.66, 0.62, 0.56, 0.49, 0.43, 0.39, 0.36],
                    "evi": [0.33, 0.36, 0.39, 0.43, 0.46, 0.49, 0.45, 0.40, 0.35, 0.31, 0.28, 0.26],
                    "ndmi": [0.22, 0.25, 0.28, 0.32, 0.35, 0.38, 0.34, 0.29, 0.24, 0.20, 0.17, 0.15]
                }
            }
        }


class ComparisonResponse(BaseModel):
    """
    Résultat détaillé de la comparaison entre deux sources satellite.
    
    Contient :
    - Les deux prédictions (GEE et Copernicus) avec leurs diagnostics respectifs
    - L'analyse de concordance (accord/divergence sur le diagnostic d'anomalie)
    - Les corrélations des séries brutes (coefficient de Pearson pour chaque indice)
    - Les différences moyennes absolues (MAE) entre les deux séries
    - Un verdict synthétique avec niveau de confiance et recommandation d'action
    
    Interprétation du verdict :
    - "Excellent accord" + Confiance "Très élevée" → Diagnostic fiable, action possible
    - "Divergence significative" + Confiance "Faible" → Vérification terrain nécessaire
    """
    parcel_id: str = Field(..., example="P0042")
    
    # Prédictions des deux sources
    prediction_gee: PredictionSource
    prediction_copernicus: PredictionSource
    
    # Analyse de concordance
    concordance: bool = Field(
        ..., 
        example=True,
        description="True si les deux sources détectent le même type (anomalie ou normal)"
    )
    difference_z_score: float = Field(
        ..., 
        example=0.036,
        description="Valeur absolue de la différence entre les deux Z-scores"
    )
    
    # Corrélations des séries brutes (Pearson)
    correlation_ndvi: float = Field(..., example=0.998, ge=-1, le=1)
    correlation_evi: float = Field(..., example=0.997, ge=-1, le=1)
    correlation_ndmi: float = Field(..., example=0.996, ge=-1, le=1)
    
    # Différences moyennes absolues (MAE)
    mae_ndvi: float = Field(..., example=0.0083, description="Erreur moyenne absolue NDVI")
    mae_evi: float = Field(..., example=0.0067, description="Erreur moyenne absolue EVI")
    mae_ndmi: float = Field(..., example=0.0058, description="Erreur moyenne absolue NDMI")
    
    # Synthèse et recommandation
    verdict: str = Field(
        ..., 
        example="Excellent accord entre les deux sources",
        description="Évaluation qualitative de la concordance"
    )
    confiance: str = Field(
        ..., 
        example="Très élevée",
        description="Niveau de confiance : Très élevée, Élevée, Modérée, Faible, Très faible"
    )
    qualite_correlation: str = Field(
        ..., 
        example="Excellente (> 0.9)",
        description="Qualité de la corrélation moyenne entre les séries"
    )
    recommandation: str = Field(
        ..., 
        example="Les deux sources confirment le diagnostic. Confiance maximale.",
        description="Recommandation d'action basée sur la concordance"
    )