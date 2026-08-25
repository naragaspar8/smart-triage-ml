from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preparar_dados


# =========================
# HIPERPARÂMETROS
# =========================
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
MAX_DEPTH = 3
RANDOM_STATE_MODELO = 42


def criar_modelo() -> GradientBoostingClassifier:
    """Cria o classificador Gradient Boosting utilizado no experimento."""
    return GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE_MODELO,
    )


def main() -> None:
    (
        X_treino,
        X_teste,
        y_treino,
        y_teste,
        label_encoder,
    ) = preparar_dados()

    modelo_gb = criar_modelo()

    modelo_gb.fit(X_treino, y_treino)
    y_pred_gb = modelo_gb.predict(X_teste)

    acuracia_gb = accuracy_score(y_teste, y_pred_gb)
    matriz_gb = confusion_matrix(y_teste, y_pred_gb)
    relatorio_gb = classification_report(
        y_teste,
        y_pred_gb,
        target_names=label_encoder.classes_,
    )

    print("\n=== GRADIENT BOOSTING ===")
    print(f"Classes codificadas: {list(label_encoder.classes_)}")
    print(f"Acurácia: {acuracia_gb:.3f}")

    print("\nMatriz de confusão:")
    print(matriz_gb)

    print("\nRelatório de classificação:")
    print(relatorio_gb)


if __name__ == "__main__":
    main()