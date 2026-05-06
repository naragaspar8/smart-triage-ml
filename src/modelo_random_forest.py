import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LEITURA DATASET
df = pd.read_csv("dataset_triagem_sintetico.csv")

print(df.head())
print(df.shape)
print(df.columns)

# EXCLUINDO COLUNA RISCO DE X E PEGANDO EM Y
x = df.drop("risco", axis=1)
y = df["risco"]

# X = ENTRADAS / Y = RESPOSTAS
print(x.head())
print(y.head())
print(type(x))
print(type(y))

# TRANSFORMA LABEL (RISCO) EM NUMERICO
label_encoder = LabelEncoder()
y_codificado = label_encoder.fit_transform(y)

print(y.head())
print(y_codificado[:5])
print(label_encoder.classes_)

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y_codificado,
    test_size=0.2,
    random_state=42,
    stratify=y_codificado
)

print(x_treino.shape)
print(x_teste.shape)
print(len(y_treino))
print(len(y_teste))

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo_rf.fit(x_treino, y_treino)

y_pred_rf = modelo_rf.predict(x_teste)

print(y_pred_rf[:10])


# =========================
# MÉTRICAS
# =========================
acuracia_rf = accuracy_score(y_teste, y_pred_rf)
matriz_rf = confusion_matrix(y_teste, y_pred_rf)
relatorio_rf = classification_report(
    y_teste,
    y_pred_rf,
    target_names=label_encoder.classes_
)

print("Acurácia Random Forest:", acuracia_rf)
print("\nMatriz de confusão Random Forest:")
print(matriz_rf)
print("\nRelatório de classificação Random Forest:")
print(relatorio_rf)