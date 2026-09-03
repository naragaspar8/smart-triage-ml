import json

import pandas as pd

from model_loader import carregar_artefato


ARTEFATO = carregar_artefato()

MODELO = ARTEFATO["model"]
LABEL_ENCODER = ARTEFATO["label_encoder"]
FEATURES = ARTEFATO["features"]
METADATA = ARTEFATO["metadata"]


def prever_risco(dados: dict) -> dict:
    campos_faltantes = [
        feature
        for feature in FEATURES
        if feature not in dados
    ]

    if campos_faltantes:
        raise ValueError(
            "Campos obrigatórios ausentes: "
            f"{campos_faltantes}"
        )

    registro = pd.DataFrame(
        [
            {
                feature: dados[feature]
                for feature in FEATURES
            }
        ]
    )

    codigo_predito = MODELO.predict(registro)[0]

    risco_predito = LABEL_ENCODER.inverse_transform(
        [codigo_predito]
    )[0]

    probabilidades_modelo = MODELO.predict_proba(
        registro
    )[0]

    classes_codificadas = MODELO.classes_

    classes = LABEL_ENCODER.inverse_transform(
        classes_codificadas
    )

    probabilidades = {
        classe: float(probabilidade)
        for classe, probabilidade
        in zip(classes, probabilidades_modelo)
    }

    return {
        "risco": risco_predito,
        "probabilidades": probabilidades,
        "modelo": METADATA["model_name"],
        "versao_modelo": METADATA["model_version"],
    }


def main() -> None:
    atendimento = {
        "idade": 72,
        "temperatura": 39.0,
        "frequencia_cardiaca": 145,
        "saturacao_oxigenio": 82,
        "pressao_sistolica": 90,
        "dor": 9,
        "dispneia": 1,
        "febre": 1,
        "comorbidades": 2,
        "tempo_espera_min": 15,
        "sangramento": 1,
        "consciencia_alterada": 1,
        "mobilidade_reduzida": 1,
    }

    resultado = prever_risco(atendimento)

    print("\n=== PREDIÇÃO ===")
    print(
        json.dumps(
            resultado,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()