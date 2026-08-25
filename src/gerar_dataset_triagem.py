import random
from pathlib import Path

import pandas as pd


TOTAL_REGISTROS = 5000
SEED = 42

PASTA_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_SAIDA = PASTA_PROJETO / "data" / "dataset_triagem_sintetico.csv"

random.seed(SEED)


def gerar_inteiro(valor_min: int, valor_max: int) -> int:
    return random.randint(valor_min, valor_max)


def gerar_decimal(valor_min: float, valor_max: float, casas: int = 1) -> float:
    return round(random.uniform(valor_min, valor_max), casas)


def gerar_binario(probabilidade_1: float) -> int:
    if not 0 <= probabilidade_1 <= 1:
        raise ValueError("A probabilidade deve estar entre 0 e 1.")

    return 1 if random.random() < probabilidade_1 else 0


def gerar_paciente_baixo() -> dict:
    idade = gerar_inteiro(0, 85)
    temperatura = gerar_decimal(35.8, 37.7)
    frequencia_cardiaca = gerar_inteiro(55, 105)
    saturacao_oxigenio = gerar_inteiro(95, 100)
    pressao_sistolica = gerar_inteiro(95, 145)
    dor = gerar_inteiro(0, 5)
    dispneia = gerar_binario(0.08)
    febre = gerar_binario(0.10)
    comorbidades = random.choices([0, 1, 2], weights=[0.65, 0.25, 0.10])[0]
    tempo_espera_min = gerar_inteiro(5, 240)
    sangramento = gerar_binario(0.02)
    consciencia_alterada = gerar_binario(0.01)
    mobilidade_reduzida = gerar_binario(0.05)

    if idade > 70 and comorbidades == 0 and random.random() < 0.30:
        comorbidades = 1

    return {
        "idade": idade,
        "temperatura": temperatura,
        "frequencia_cardiaca": frequencia_cardiaca,
        "saturacao_oxigenio": saturacao_oxigenio,
        "pressao_sistolica": pressao_sistolica,
        "dor": dor,
        "dispneia": dispneia,
        "febre": febre,
        "comorbidades": comorbidades,
        "tempo_espera_min": tempo_espera_min,
        "sangramento": sangramento,
        "consciencia_alterada": consciencia_alterada,
        "mobilidade_reduzida": mobilidade_reduzida,
        "risco": "baixo",
    }


def gerar_paciente_medio() -> dict:
    idade = gerar_inteiro(10, 95)
    temperatura = gerar_decimal(36.4, 38.6)
    frequencia_cardiaca = gerar_inteiro(72, 115)
    saturacao_oxigenio = gerar_inteiro(91, 97)
    pressao_sistolica = gerar_inteiro(90, 160)
    dor = gerar_inteiro(3, 7)
    dispneia = gerar_binario(0.25)
    febre = gerar_binario(0.30)
    comorbidades = random.choices([0, 1, 2], weights=[0.40, 0.40, 0.20])[0]
    tempo_espera_min = gerar_inteiro(15, 210)
    sangramento = gerar_binario(0.06)
    consciencia_alterada = gerar_binario(0.03)
    mobilidade_reduzida = gerar_binario(0.12)

    if dispneia == 1 and saturacao_oxigenio > 95 and random.random() < 0.50:
        saturacao_oxigenio = gerar_inteiro(90, 95)

    if febre == 1 and temperatura < 37.6 and random.random() < 0.70:
        temperatura = gerar_decimal(37.6, 38.5)

    return {
        "idade": idade,
        "temperatura": temperatura,
        "frequencia_cardiaca": frequencia_cardiaca,
        "saturacao_oxigenio": saturacao_oxigenio,
        "pressao_sistolica": pressao_sistolica,
        "dor": dor,
        "dispneia": dispneia,
        "febre": febre,
        "comorbidades": comorbidades,
        "tempo_espera_min": tempo_espera_min,
        "sangramento": sangramento,
        "consciencia_alterada": consciencia_alterada,
        "mobilidade_reduzida": mobilidade_reduzida,
        "risco": "medio",
    }


def gerar_paciente_alto() -> dict:
    idade = gerar_inteiro(0, 100)
    temperatura = gerar_decimal(36.8, 40.2)
    frequencia_cardiaca = gerar_inteiro(100, 180)
    saturacao_oxigenio = gerar_inteiro(70, 93)
    pressao_sistolica = gerar_inteiro(70, 220)
    dor = gerar_inteiro(7, 10)
    dispneia = gerar_binario(0.65)
    febre = gerar_binario(0.40)
    comorbidades = random.choices([0, 1, 2], weights=[0.15, 0.35, 0.50])[0]
    tempo_espera_min = gerar_inteiro(0, 180)
    sangramento = gerar_binario(0.35)
    consciencia_alterada = gerar_binario(0.25)
    mobilidade_reduzida = gerar_binario(0.40)

    if consciencia_alterada == 1 and saturacao_oxigenio > 92 and random.random() < 0.60:
        saturacao_oxigenio = gerar_inteiro(75, 92)

    if dispneia == 1 and saturacao_oxigenio > 90 and random.random() < 0.60:
        saturacao_oxigenio = gerar_inteiro(70, 90)

    if sangramento == 1 and pressao_sistolica > 110 and random.random() < 0.50:
        pressao_sistolica = gerar_inteiro(70, 110)

    return {
        "idade": idade,
        "temperatura": temperatura,
        "frequencia_cardiaca": frequencia_cardiaca,
        "saturacao_oxigenio": saturacao_oxigenio,
        "pressao_sistolica": pressao_sistolica,
        "dor": dor,
        "dispneia": dispneia,
        "febre": febre,
        "comorbidades": comorbidades,
        "tempo_espera_min": tempo_espera_min,
        "sangramento": sangramento,
        "consciencia_alterada": consciencia_alterada,
        "mobilidade_reduzida": mobilidade_reduzida,
        "risco": "alto",
    }


def gerar_registro_por_risco(risco: str) -> dict:
    if risco == "baixo":
        return gerar_paciente_baixo()
    if risco == "medio":
        return gerar_paciente_medio()
    if risco == "alto":
        return gerar_paciente_alto()

    raise ValueError(f"Risco inválido: {risco}")


def gerar_dataset(total_registros: int) -> pd.DataFrame:
    riscos = random.choices(
        population=["baixo", "medio", "alto"],
        weights=[0.50, 0.30, 0.20],
        k=total_registros,
    )

    registros = [gerar_registro_por_risco(risco) for risco in riscos]
    df = pd.DataFrame(registros)

    df["idade"] = df["idade"].clip(0, 100)
    df["temperatura"] = df["temperatura"].clip(34.0, 41.0)
    df["frequencia_cardiaca"] = df["frequencia_cardiaca"].clip(40, 180)
    df["saturacao_oxigenio"] = df["saturacao_oxigenio"].clip(70, 100)
    df["pressao_sistolica"] = df["pressao_sistolica"].clip(70, 220)
    df["dor"] = df["dor"].clip(0, 10)
    df["comorbidades"] = df["comorbidades"].clip(0, 2)
    df["tempo_espera_min"] = df["tempo_espera_min"].clip(0, 720)
    df["sangramento"] = df["sangramento"].clip(0, 1)
    df["consciencia_alterada"] = df["consciencia_alterada"].clip(0, 1)
    df["dispneia"] = df["dispneia"].clip(0, 1)
    df["febre"] = df["febre"].clip(0, 1)
    df["mobilidade_reduzida"] = df["mobilidade_reduzida"].clip(0, 1)

    return df


def validar_dataset(df: pd.DataFrame, total_esperado: int) -> None:
    if len(df) != total_esperado:
        raise ValueError(
            f"Quantidade de registros inválida: {len(df)}. "
            f"Esperado: {total_esperado}."
        )

    if df.isnull().any().any():
        raise ValueError("O dataset contém valores nulos.")

    if "risco" not in df.columns:
        raise ValueError("A coluna alvo 'risco' não foi encontrada.")

    classes_validas = {"baixo", "medio", "alto"}
    classes_encontradas = set(df["risco"].unique())

    if not classes_encontradas.issubset(classes_validas):
        raise ValueError(
            f"Foram encontradas classes inválidas: {classes_encontradas}"
        )


def imprimir_resumo(df: pd.DataFrame) -> None:
    print("\n=== TAMANHO DO DATASET ===")
    print(df.shape)

    print("\n=== DISTRIBUIÇÃO DA CLASSE RISCO ===")
    print(df["risco"].value_counts())

    print("\n=== DISTRIBUIÇÃO PERCENTUAL ===")
    print((df["risco"].value_counts(normalize=True) * 100).round(2))

    print("\n=== DUPLICIDADES ===")
    print(df.duplicated().sum())

    print("\n=== MÉDIAS POR CLASSE ===")
    colunas_numericas = [
        "idade",
        "temperatura",
        "frequencia_cardiaca",
        "saturacao_oxigenio",
        "pressao_sistolica",
        "dor",
        "comorbidades",
        "tempo_espera_min",
        "dispneia",
        "febre",
        "sangramento",
        "consciencia_alterada",
        "mobilidade_reduzida",
    ]

    print(df.groupby("risco")[colunas_numericas].mean().round(2))


def main() -> None:
    df = gerar_dataset(TOTAL_REGISTROS)

    validar_dataset(df, TOTAL_REGISTROS)
    imprimir_resumo(df)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")

    print(f"\nArquivo salvo com sucesso: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()