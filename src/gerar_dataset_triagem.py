import random
import pandas as pd


# DEFININDO CONSTANTES DE CONFIGURACOES GERAIS 
TOTAL_REGISTROS = 5000
ARQUIVO_SAIDA = "dataset_triagem_sintetico.csv"
SEED = 42

random.seed(SEED)

# =========================
# FUNÇÕES AUXILIARES
# =========================
def gerar_inteiro(valor_min: int, valor_max: int) -> int:
    return random.randint(valor_min,valor_max)

def gerar_decimal(valor_min: float, valor_max: float, casas: int = 1) -> float:
    return round(random.uniform(valor_min,valor_max), casas)

def gerar_binario(probabilidade_1: float) -> int:
    """
    Retorna 1 com a probabilidade informada, senão 0.
    Exemplo: gerar_binario(0.2) => 20% de chance de retornar 1
    """
    return 1 if random.random() < probabilidade_1 else 0
    
def limitar_valor(valor, minimo, maximo):
    return max(minimo, min (valor, maximo))

# =========================
# GERAÇÃO POR CLASSE
# =========================
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

    # Pequenos ruídos plausíveis
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

    # Ajustes de coerência simples
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

    # Ajustes de coerência
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


# =========================
# GERAÇÃO GERAL DO DATASET
# =========================
def gerar_registro_por_risco(risco: str) -> dict:
    if risco == "baixo":
        return gerar_paciente_baixo()
    elif risco == "medio":
        return gerar_paciente_medio()
    elif risco == "alto":
        return gerar_paciente_alto()
    else:
        raise ValueError(f"Risco inválido: {risco}")


def gerar_dataset(total_registros: int) -> pd.DataFrame:
    riscos = random.choices(
        population=["baixo", "medio", "alto"],
        weights=[0.50, 0.30, 0.20],
        k=total_registros,
    )

    registros = [gerar_registro_por_risco(risco) for risco in riscos]
    df = pd.DataFrame(registros)

    # Garantia extra de limites
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


# =========================
# RELATÓRIO BÁSICO
# =========================
def imprimir_resumo(df: pd.DataFrame) -> None:
    print("\n=== TAMANHO DO DATASET ===")
    print(df.shape)

    print("\n=== DISTRIBUIÇÃO DA CLASSE RISCO ===")
    print(df["risco"].value_counts())
    print("\n=== DISTRIBUIÇÃO PERCENTUAL ===")
    print((df["risco"].value_counts(normalize=True) * 100).round(2))

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


# =========================
# EXECUÇÃO PRINCIPAL
# =========================
if __name__ == "__main__":
    df = gerar_dataset(TOTAL_REGISTROS)
    imprimir_resumo(df)
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    print(f"\nArquivo salvo com sucesso: {ARQUIVO_SAIDA}") 