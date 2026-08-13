from pydantic import BaseModel, Field
from typing import List


class SerieTemporelle(BaseModel):
    """
    Une série temporelle NDVI/EVI/NDMI pour UNE parcelle.
    Les 3 listes doivent avoir la même longueur (nombre de dates),
    idéalement 12 points (une année, comme le jeu de test utilisé
    à l'entraînement).
    """
    parcel_id: str = Field(..., example="P0001")
    ndvi: List[float] = Field(..., example=[0.45, 0.42, 0.38, 0.30, 0.25,
                                             0.22, 0.20, 0.24, 0.30, 0.38, 0.44, 0.46])
    evi: List[float] = Field(..., example=[0.30, 0.28, 0.25, 0.20, 0.17,
                                            0.15, 0.14, 0.16, 0.20, 0.25, 0.29, 0.31])
    ndmi: List[float] = Field(..., example=[0.10, 0.08, 0.05, -0.02, -0.08,
                                             -0.12, -0.14, -0.10, -0.03, 0.04, 0.09, 0.11])


class ResultatPrediction(BaseModel):
    parcel_id: str
    erreur_reconstruction: float
    z_score: float
    anomalie: bool
    criticite: str
    criticite_pct: float
