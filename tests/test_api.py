from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health_deve_retornar_status_ok():
    response = client.get("/health")

    assert response.status_code == 200

    dados = response.json()

    assert dados["status"] == "ok"
    assert dados["model_loaded"] is True
    assert dados["model"] == "random_forest"
    assert dados["model_version"] == "1.0.0"


def test_predict_deve_classificar_atendimento_valido():
    atendimento = {
        "idade": 72,
        "temperatura": 39.0,
        "frequencia_cardiaca": 145,
        "saturacao_oxigenio": 82,
        "pressao_sistolica": 90,
        "dor": 9,
        "dispneia": 1,
        "febre": 1,
        "comorbidades": 2,
        "tempo_espera_min": 15,
        "sangramento": 1,
        "consciencia_alterada": 1,
        "mobilidade_reduzida": 1,
    }

    response = client.post(
        "/predict",
        json=atendimento,
    )

    assert response.status_code == 200

    dados = response.json()

    assert dados["risco"] in {
        "alto",
        "baixo",
        "medio",
    }

    assert dados["modelo"] == "random_forest"
    assert dados["versao_modelo"] == "1.0.0"

    assert "probabilidades" in dados

    probabilidades = dados["probabilidades"]

    assert set(probabilidades.keys()) == {
        "alto",
        "baixo",
        "medio",
    }

    assert abs(
        sum(probabilidades.values()) - 1.0
    ) < 0.000001


def test_predict_deve_rejeitar_dados_invalidos():
    atendimento = {
        "idade": -10,
        "temperatura": 34,
        "frequencia_cardiaca": 40,
        "saturacao_oxigenio": 150,
        "pressao_sistolica": 70,
        "dor": 10,
        "dispneia": 1,
        "febre": 1,
        "comorbidades": 2,
        "tempo_espera_min": 720,
        "sangramento": 1,
        "consciencia_alterada": 1,
        "mobilidade_reduzida": 1,
    }

    response = client.post(
        "/predict",
        json=atendimento,
    )

    assert response.status_code == 422


def test_predict_deve_rejeitar_campo_ausente():
    atendimento = {
        "idade": 72,
        "temperatura": 39.0,
        "frequencia_cardiaca": 145,
        "pressao_sistolica": 90,
        "dor": 9,
        "dispneia": 1,
        "febre": 1,
        "comorbidades": 2,
        "tempo_espera_min": 15,
        "sangramento": 1,
        "consciencia_alterada": 1,
        "mobilidade_reduzida": 1,
    }

    response = client.post(
        "/predict",
        json=atendimento,
    )

    assert response.status_code == 422