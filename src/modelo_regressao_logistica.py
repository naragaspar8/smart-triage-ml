from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import preparar_dados


MAX_ITER = 1000
SOLVER = "lbfgs"


def criar_modelo() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "modelo",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    solver=SOLVER,
                ),
            ),
        ]
    )


def main() -> None:
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
    matriz = confusion_matrix(y_teste, y_pred)
    relatorio = classification_report(
        y_teste,
        y_pred,
        target_names=label_encoder.classes_,
    )

    print("\n=== REGRESSÃO LOGÍSTICA ===")
    print(f"Classes codificadas: {list(label_encoder.classes_)}")
    print(f"Acurácia: {acuracia:.3f}")

    print("\nMatriz de confusão:")
    print(matriz)

    print("\nRelatório de classificação:")
    print(relatorio)


if __name__ == "__main__":
    main()