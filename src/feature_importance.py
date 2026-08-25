from pathlib import Path

import pandas as pd

from modelo_random_forest import criar_modelo
from preprocessing import preparar_dados


# =========================
# CONFIGURAÇÕES
# =========================
PASTA_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_RESULTADOS = (
    PASTA_PROJETO
    / "reports"
    / "metricas"
    / "feature_importance_random_forest.csv"
)


def calcular_importancia_variaveis() -> pd.DataFrame:
    """
    Treina o Random Forest com o conjunto de treino e
    extrai a importância das variáveis calculada pelo modelo.
    """
    (
        X_treino,
        _X_teste,
        y_treino,
        _y_teste,
        _label_encoder,
    ) = preparar_dados()

    modelo_rf = criar_modelo()
    modelo_rf.fit(X_treino, y_treino)

    importancias = modelo_rf.feature_importances_

    df_importancias = pd.DataFrame(
        {
            "variavel": X_treino.columns,
            "importancia": importancias,
        }
    )

    df_importancias = df_importancias.sort_values(
        by="importancia",
        ascending=False,
    ).reset_index(drop=True)

    df_importancias["posicao"] = df_importancias.index + 1

    return df_importancias[
        ["posicao", "variavel", "importancia"]
    ]


def main() -> None:
    df_importancias = calcular_importancia_variaveis()

    ARQUIVO_RESULTADOS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_importancias.to_csv(
        ARQUIVO_RESULTADOS,
        index=False,
        encoding="utf-8",
    )

    print("\n=== IMPORTÂNCIA DAS VARIÁVEIS - RANDOM FOREST ===")
    print(
        df_importancias.to_string(
            index=False,
            formatters={
                "importancia": lambda valor: f"{valor:.6f}"
            },
        )
    )

    print("\nSoma das importâncias:")
    print(f"{df_importancias['importancia'].sum():.6f}")

    print(
        f"\nResultados salvos em: {ARQUIVO_RESULTADOS}"
    )


if __name__ == "__main__":
    main()