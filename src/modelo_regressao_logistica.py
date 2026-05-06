import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LE O ARQUIVO CSV
df = pd.read_csv("dataset_triagem_sintetico.csv")

print(df.head())
print(df.shape)
print(df.columns)

# REMOVE A COLUNA 'RISCO' DA TABELA
x = df.drop("risco", axis=1)

# PEGA SOMENTE A COLUNA 'RISCO'
y = df["risco"]

print(x.head())
print(y.head())
print(type(x))
print(type(y))

# TRANFORMANDO 'RISCO' PARA NUMÉRICO
label_encoder = LabelEncoder ()
y_codificado = label_encoder.fit_transform(y)

print(y.head())
print(y_codificado[:5])
print(label_encoder.classes_)

# DIVISAO DATASET TREINO E TESTE
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


# PADRONIZACAO DE VARIAVEIS 
scaler = StandardScaler()

x_treino_escalado = scaler.fit_transform(x_treino)
x_teste_escalado = scaler.transform(x_teste)

print(x_treino_escalado[:3])
print(x_teste_escalado[:3])


# REGRESSAO LOGISTICA
modelo = LogisticRegression(max_iter=1000)
modelo.fit(x_treino_escalado, y_treino)

y_pred = modelo.predict(x_teste_escalado)

print(y_pred[:10])

# ACURÁCIA, MATRIZ E RELATÓRIO
acuracia = accuracy_score(y_teste, y_pred)
matriz = confusion_matrix(y_teste, y_pred)
relatorio = classification_report(y_teste, y_pred, target_names=label_encoder.classes_)

print("\nAcurácia: ", acuracia)
print("\nMatriz de Confusâo:")
print(matriz)
print("\nRelatório de Classificação:")
print(relatorio)