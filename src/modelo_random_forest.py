from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preparar_dados


N_ESTIMATORS = 100
RANDOM_STATE_MODELO = 42


def criar_modelo() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
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

    modelo_rf = criar_modelo()

    modelo_rf.fit(X_treino, y_treino)
    y_pred_rf = modelo_rf.predict(X_teste)

    acuracia_rf = accuracy_score(y_teste, y_pred_rf)
    matriz_rf = confusion_matrix(y_teste, y_pred_rf)
    relatorio_rf = classification_report(
        y_teste,
        y_pred_rf,
        target_names=label_encoder.classes_,
    )

    print("\n=== RANDOM FOREST ===")
    print(f"Classes codificadas: {list(label_encoder.classes_)}")
    print(f"Acurácia: {acuracia_rf:.3f}")

    print("\nMatriz de confusão:")
    print(matriz_rf)

    print("\nRelatório de classificação:")
    print(relatorio_rf)


if __name__ == "__main__":
    main()