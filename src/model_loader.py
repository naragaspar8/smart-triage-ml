from pathlib import Path

import joblib


PASTA_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = (
    PASTA_PROJETO
    / "models"
    / "modelo_triagem_v1.joblib"
)

CHAVES_OBRIGATORIAS = {
    "model",
    "label_encoder",
    "features",
    "metadata",
}


def carregar_artefato():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {CAMINHO_MODELO}"
        )

    artefato = joblib.load(CAMINHO_MODELO)

    chaves_faltantes = (
        CHAVES_OBRIGATORIAS - artefato.keys()
    )

    if chaves_faltantes:
        raise ValueError(
            "Artefato inválido. "
            f"Chaves ausentes: {chaves_faltantes}"
        )

    return artefato