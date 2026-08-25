from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PASTA_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_DATASET = PASTA_PROJETO / "data" / "dataset_triagem_sintetico.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def carregar_dataset() -> pd.DataFrame:
    if not ARQUIVO_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {ARQUIVO_DATASET}"
        )

    return pd.read_csv(ARQUIVO_DATASET)


def preparar_dados():
    """
    Executa o pré-processamento comum aos modelos:
    1. carrega o dataset;
    2. separa X e y;
    3. codifica a variável alvo;
    4. divide em treino e teste de forma estratificada.
    """
    df = carregar_dataset()

    X = df.drop("risco", axis=1)
    y = df["risco"]

    label_encoder = LabelEncoder()
    y_codificado = label_encoder.fit_transform(y)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y_codificado,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_codificado,
    )

    return (
        X_treino,
        X_teste,
        y_treino,
        y_teste,
        label_encoder,
    )