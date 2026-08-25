from pathlib import Path

import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from modelo_gradient_boosting import criar_modelo as criar_gradient_boosting
from modelo_random_forest import criar_modelo as criar_random_forest
from modelo_regressao_logistica import criar_modelo as criar_regressao_logistica
from preprocessing import preparar_dados


# =========================
# CONFIGURAÇÕES
# =========================
N_SPLITS = 5
RANDOM_STATE_CV = 42

PASTA_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_RESULTADOS = (
    PASTA_PROJETO
    / "reports"
    / "metricas"
    / "validacao_cruzada.csv"
)


def criar_metricas(indice_alto: int) -> dict:
    """
    Define as métricas utilizadas na validação cruzada.

    Precision, recall e F1 são calculados usando média macro,
    dando o mesmo peso para cada classe.

    O recall da classe alto risco é calculado separadamente,
    usando o código obtido dinamicamente pelo LabelEncoder.
    """
    recall_alto = make_scorer(
        recall_score,
        labels=[indice_alto],
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "recall_alto": recall_alto,
    }


def avaliar_modelo(
    nome_modelo: str,
    modelo,
    X_treino,
    y_treino,
    validacao,
    metricas,
) -> dict:
    """
    Executa a validação cruzada de um modelo e retorna
    as médias e desvios padrão das métricas.
    """
    resultados = cross_validate(
        estimator=modelo,
        X=X_treino,
        y=y_treino,
        cv=validacao,
        scoring=metricas,
        return_train_score=False,
    )

    print(f"\n=== {nome_modelo.upper()} ===")

    for numero_fold in range(N_SPLITS):
        print(
            f"Fold {numero_fold + 1}: "
            f"accuracy={resultados['test_accuracy'][numero_fold]:.4f} | "
            f"precision_macro={resultados['test_precision_macro'][numero_fold]:.4f} | "
            f"recall_macro={resultados['test_recall_macro'][numero_fold]:.4f} | "
            f"f1_macro={resultados['test_f1_macro'][numero_fold]:.4f} | "
            f"recall_alto={resultados['test_recall_alto'][numero_fold]:.4f}"
        )

    resumo = {
        "modelo": nome_modelo,
        "accuracy_media": resultados["test_accuracy"].mean(),
        "accuracy_desvio": resultados["test_accuracy"].std(),
        "precision_macro_media": resultados["test_precision_macro"].mean(),
        "precision_macro_desvio": resultados["test_precision_macro"].std(),
        "recall_macro_media": resultados["test_recall_macro"].mean(),
        "recall_macro_desvio": resultados["test_recall_macro"].std(),
        "f1_macro_media": resultados["test_f1_macro"].mean(),
        "f1_macro_desvio": resultados["test_f1_macro"].std(),
        "recall_alto_media": resultados["test_recall_alto"].mean(),
        "recall_alto_desvio": resultados["test_recall_alto"].std(),
    }

    print("\nResumo:")
    print(
        f"Accuracy: {resumo['accuracy_media']:.4f} "
        f"(± {resumo['accuracy_desvio']:.4f})"
    )
    print(
        f"Precision macro: {resumo['precision_macro_media']:.4f} "
        f"(± {resumo['precision_macro_desvio']:.4f})"
    )
    print(
        f"Recall macro: {resumo['recall_macro_media']:.4f} "
        f"(± {resumo['recall_macro_desvio']:.4f})"
    )
    print(
        f"F1 macro: {resumo['f1_macro_media']:.4f} "
        f"(± {resumo['f1_macro_desvio']:.4f})"
    )
    print(
        f"Recall alto: {resumo['recall_alto_media']:.4f} "
        f"(± {resumo['recall_alto_desvio']:.4f})"
    )

    return resumo


def main() -> None:
    (
        X_treino,
        _X_teste,
        y_treino,
        _y_teste,
        label_encoder,
    ) = preparar_dados()

    print("=== VALIDAÇÃO CRUZADA ESTRATIFICADA ===")
    print(f"Número de folds: {N_SPLITS}")
    print(f"Random state: {RANDOM_STATE_CV}")
    print(f"Registros utilizados na validação cruzada: {len(X_treino)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(
        "Observação: o conjunto de teste de 20% foi preservado "
        "e não participa desta validação cruzada."
    )

    validacao = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE_CV,
    )

    indice_alto = int(label_encoder.transform(["alto"])[0])
    metricas = criar_metricas(indice_alto)

    modelos = {
        "Regressão Logística": criar_regressao_logistica(),
        "Random Forest": criar_random_forest(),
        "Gradient Boosting": criar_gradient_boosting(),
    }

    resumos = []

    for nome_modelo, modelo in modelos.items():
        resumo = avaliar_modelo(
            nome_modelo=nome_modelo,
            modelo=modelo,
            X_treino=X_treino,
            y_treino=y_treino,
            validacao=validacao,
            metricas=metricas,
        )
        resumos.append(resumo)

    df_resultados = pd.DataFrame(resumos)

    ARQUIVO_RESULTADOS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_resultados.to_csv(
        ARQUIVO_RESULTADOS,
        index=False,
        encoding="utf-8",
    )

    print("\n=== COMPARAÇÃO FINAL ===")
    colunas_exibicao = [
        "modelo",
        "accuracy_media",
        "accuracy_desvio",
        "recall_macro_media",
        "f1_macro_media",
        "recall_alto_media",
    ]

    print(
        df_resultados[colunas_exibicao]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nResultados salvos em: {ARQUIVO_RESULTADOS}"
    )


if __name__ == "__main__":
    main()