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
