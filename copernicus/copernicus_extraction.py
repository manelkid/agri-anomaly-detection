"""
Extraction NDVI / EVI / NDMI via Copernicus Data Space Ecosystem
Sentinel-2 L2A - Statistical API
"""

import json
import requests
import pandas as pd
from sentinelhub import SHConfig, SentinelHubSession
from dotenv import load_dotenv
import os
import pathlib
# Racine du projet agri-anomaly-detection
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Fichier .env
ENV_FILE = BASE_DIR / "config" / ".env"

# Charger les variables
print("📁 Recherche du fichier .env :", ENV_FILE)
print("📁 Fichier .env existe :", ENV_FILE.exists())

load_dotenv(dotenv_path=ENV_FILE, override=True)

# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")

print("🔑 Client ID chargé :", bool(CLIENT_ID))
print("🔐 Client Secret chargé :", bool(CLIENT_SECRET))

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "\n❌ Identifiants Copernicus introuvables.\n"
        f"Fichier recherché : {ENV_FILE}\n"
        f"Existe : {ENV_FILE.exists()}\n\n"
        "Le fichier .env doit contenir :\n"
        "COPERNICUS_CLIENT_ID=...\n"
        "COPERNICUS_CLIENT_SECRET=...\n"
    )

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET

config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET

config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

DATE_DEBUT = "2025-01-01"
DATE_FIN = "2025-12-31"

NB_PARCELLES_MAX = None

CHEMIN_GEOJSON = "data/Beja_parcelles_800.geojson"


# ============================================================
# ÉVALSCRIPT
# ============================================================

EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: [
                "B02",
                "B04",
                "B08",
                "B11",
                "SCL",
                "dataMask"
            ],

            units: [
                "REFLECTANCE",
                "REFLECTANCE",
                "REFLECTANCE",
                "REFLECTANCE",
                "DN",
                "DN"
            ]
        }],

        output: [
            {
                id: "indices",
                bands: 3,
                sampleType: "FLOAT32"
            },
            {
                id: "dataMask",
                bands: 1
            }
        ]
    };
}


function evaluatePixel(samples) {

    // ==========================
    // NDVI
    // ==========================

    var ndvi =
        (samples.B08 - samples.B04) /
        (samples.B08 + samples.B04 + 0.0001);


    // ==========================
    // EVI
    // ==========================

    var evi =
        2.5 *
        (
            (samples.B08 - samples.B04) /
            (
                samples.B08
                + 6.0 * samples.B04
                - 7.5 * samples.B02
                + 1.0
            )
        );


    // ==========================
    // NDMI
    // ==========================

    var ndmi =
        (samples.B08 - samples.B11) /
        (samples.B08 + samples.B11 + 0.0001);


    // ==========================
    // MASQUE NUAGES
    // ==========================

    var cloud =
        samples.SCL == 3 ||
        samples.SCL == 8 ||
        samples.SCL == 9 ||
        samples.SCL == 10 ||
        samples.SCL == 11;


    // ==========================
    // PIXEL NON VALIDE
    // ==========================

    if (cloud) {

        return {
            indices: [0, 0, 0],
            dataMask: [0]
        };

    }


    // ==========================
    // PIXEL VALIDE
    // ==========================

    return {
        indices: [
            ndvi,
            evi,
            ndmi
        ],

        dataMask: [
            samples.dataMask
        ]
    };
}
"""

# ============================================================
# CHARGEMENT DES PARCELLES
# ============================================================

def charger_parcelles_depuis_geojson(
    chemin: str,
    limite: int = None
) -> dict:

    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)

    parcelles = {}

    for feature in data["features"]:

        props = feature["properties"]

        parcel_id = props["parcel_id"]

        bbox = [
            props["lon_min"],
            props["lat_min"],
            props["lon_max"],
            props["lat_max"]
        ]

        parcelles[parcel_id] = bbox

    if limite:
        parcelles = dict(
            list(parcelles.items())[:limite]
        )

    print(
        f"Parcelles chargées depuis "
        f"{chemin} : {len(parcelles)}"
    )

    return parcelles


PARCELLES_BBOX = charger_parcelles_depuis_geojson(
    CHEMIN_GEOJSON,
    NB_PARCELLES_MAX
)


# ============================================================
# AUTHENTIFICATION
# ============================================================

def obtenir_token() -> str:

    print("🔐 Authentification...")

    session = SentinelHubSession(
        config=config,
        refresh_before_expiry=120
    )

    token_info = session.token

    print(
        "Type du token :",
        type(token_info)
    )

    print(
        "Clés du token :",
        token_info.keys()
    )

    access_token = token_info["access_token"]

    print(
        "Token valide :",
        access_token[:20] + "..."
    )

    return access_token


# ============================================================
# EXTRACTION D'UNE PARCELLE
# ============================================================

def extraire_parcelle(
    parcel_id: str,
    bbox_coords: list,
    token: str
) -> pd.DataFrame:

    url = (
        "https://sh.dataspace.copernicus.eu/"
        "statistics/v1"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


    # --------------------------------------------------------
    # PAYLOAD
    # --------------------------------------------------------

    payload = {

        "input": {

            "bounds": {

                "bbox": bbox_coords,

                "properties": {
                    "crs": (
                        "http://www.opengis.net/"
                        "def/crs/EPSG/0/4326"
                    )
                }
            },

            "data": [

                {
                    "type": "sentinel-2-l2a",

                    "dataFilter": {

                        "mosaickingOrder": "leastCC"

                    }
                }

            ]
        },


        "aggregation": {

            "timeRange": {

                "from": (
                    f"{DATE_DEBUT}"
                    "T00:00:00Z"
                ),

                "to": (
                    f"{DATE_FIN}"
                    "T23:59:59Z"
                )
            },


            "aggregationInterval": {

                "of": "P1M"

            },


            "evalscript": EVALSCRIPT,


            "resx": 10,

            "resy": 10
        },


        "calculations": {

            "default": {}

        }
    }


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print(
        f"  📡 Requête pour {parcel_id}"
    )

    print(
        f"  BBOX : {bbox_coords}"
    )


    # --------------------------------------------------------
    # APPEL API
    # --------------------------------------------------------

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

    except requests.RequestException as e:

        print(
            f"  ❌ Erreur réseau : {e}"
        )

        return pd.DataFrame()


    # --------------------------------------------------------
    # ERREUR API
    # --------------------------------------------------------

    if response.status_code != 200:

        print(
            f"  ❌ Erreur HTTP "
            f"{response.status_code}"
        )

        print(
            f"  Réponse : {response.text}"
        )

        return pd.DataFrame()


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        print(
            "  ❌ Réponse non JSON"
        )

        print(response.text)

        return pd.DataFrame()


    # --------------------------------------------------------
    # AFFICHAGE STRUCTURE
    # --------------------------------------------------------

    print(
        "  ✅ Réponse reçue"
    )

    print(
        "  Nombre d'intervalles :",
        len(data.get("data", []))
    )


    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    lignes = []


    for entree in data.get("data", []):

        interval = entree.get(
            "interval",
            {}
        )

        date = interval.get(
            "from",
            ""
        )[:10]


        outputs = entree.get(
            "outputs",
            {}
        )


        # Notre output s'appelle "indices"
        indices_output = outputs.get(
            "indices"
        )


        if not indices_output:

            continue


        bands = indices_output.get(
            "bands",
            {}
        )


        # ----------------------------------------------------
        # IMPORTANT :
        # Les statistiques sont normalement B0/B1/B2
        # ----------------------------------------------------

        ndvi_mean = (
            bands
            .get("B0", {})
            .get("stats", {})
            .get("mean")
        )

        evi_mean = (
            bands
            .get("B1", {})
            .get("stats", {})
            .get("mean")
        )

        ndmi_mean = (
            bands
            .get("B2", {})
            .get("stats", {})
            .get("mean")
        )


        # Ajouter seulement si au moins
        # une valeur existe

        if (
            ndvi_mean is not None
            or evi_mean is not None
            or ndmi_mean is not None
        ):

            lignes.append({

                "parcel_id": parcel_id,

                "date": date,

                "NDVI": ndvi_mean,

                "EVI": evi_mean,

                "NDMI": ndmi_mean

            })


    return pd.DataFrame(lignes)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # AUTHENTIFICATION
    # --------------------------------------------------------

    token = obtenir_token()

    print("✅ Token obtenu\n")


    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    toutes_les_lignes = []


    for i, (
        parcel_id,
        bbox_coords
    ) in enumerate(
        PARCELLES_BBOX.items(),
        1
    ):

        print(
            f"[{i}/{len(PARCELLES_BBOX)}] "
            f"Extraction {parcel_id}..."
        )


        try:

            df_parcelle = extraire_parcelle(
                parcel_id,
                bbox_coords,
                token
            )


            if not df_parcelle.empty:

                toutes_les_lignes.append(
                    df_parcelle
                )

                print(
                    f"  ✅ "
                    f"{len(df_parcelle)} "
                    f"observations extraites"
                )

            else:

                print(
                    "  ⚠️ Aucune donnée"
                )


        except Exception as e:

            print(
                f"  ❌ Erreur : {e}"
            )

            continue


    # --------------------------------------------------------
    # SAUVEGARDE
    # --------------------------------------------------------

    if toutes_les_lignes:

        df_final = pd.concat(
            toutes_les_lignes,
            ignore_index=True
        )


        df_final.to_csv(
            "data/copernicus_donnees.csv",
            index=False
        )


        print(
            "\n✅ Extraction terminée"
        )

        print(
            f"📁 {len(df_final)} lignes "
            "-> copernicus_donnees.csv"
        )


        print(
            "\n📊 Aperçu :"
        )

        print(
            df_final.head(15).to_string(
                index=False
            )
        )


    else:

        print(
            "\n❌ Aucune donnée extraite"
        )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    main()