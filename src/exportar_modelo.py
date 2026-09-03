from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from preprocessing import preparar_dados
from modelo_random_forest import (
    criar_modelo,
    N_ESTIMATORS,
    RANDOM_STATE_MODELO,
)


PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_MODELOS = PASTA_PROJETO / "models"

NOME_ARQUIVO_MODELO = "modelo_triagem_v1.joblib"
CAMINHO_MODELO = PASTA_MODELOS / NOME_ARQUIVO_MODELO

VERSAO_MODELO = "1.0.0"


def exportar_modelo() -> None:
    (
        X_treino,
        X_teste,
        y_treino,
        y_teste,
        label_encoder,
    ) = preparar_dados()

    modelo = criar_modelo()

    modelo.fit(X_treino, y_treino)

    y_pred = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, y_pred)
    precisao_macro = precision_score(
        y_teste,
        y_pred,
        average="macro",
    )
    recall_macro = recall_score(
        y_teste,
        y_pred,
        average="macro",
    )
    f1_macro = f1_score(
        y_teste,
        y_pred,
        average="macro",
    )

    codigo_alto_risco = label_encoder.transform(["alto"])[0]

    recall_alto_risco = recall_score(
        y_teste == codigo_alto_risco,
        y_pred == codigo_alto_risco,
    )

    artefato = {
        "model": modelo,
        "label_encoder": label_encoder,
        "features": list(X_treino.columns),
        "metadata": {
            "model_name": "random_forest",
            "model_version": VERSAO_MODELO,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE_MODELO,
            "training_records": len(X_treino),
            "test_records": len(X_teste),
            "metrics": {
                "accuracy": acuracia,
                "precision_macro": precisao_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "recall_alto_risco": recall_alto_risco,
            },
        },
    }

    PASTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artefato,
        CAMINHO_MODELO,
    )

    print("\n=== MODELO EXPORTADO ===")
    print(f"Arquivo: {CAMINHO_MODELO}")
    print(f"Versão: {VERSAO_MODELO}")
    print(f"Features: {len(X_treino.columns)}")
    print(f"Acurácia: {acuracia:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"Recall alto risco: {recall_alto_risco:.4f}")


if __name__ == "__main__":
    exportar_modelo()