from typing import Literal
from pydantic import BaseModel, Field

class TriagemRequest(BaseModel):
    idade: int = Field(ge=0, le=100)
    temperatura: float = Field(ge=34, le=41)
    frequencia_cardiaca: int = Field(ge=40, le=180)
    saturacao_oxigenio: int = Field(ge=70, le=100)
    pressao_sistolica: int = Field(ge=70, le=220)
    dor: int = Field(ge=0, le=10)
    dispneia: int = Field(ge=0, le=1)
    febre: int = Field(ge=0, le=1)
    comorbidades: int = Field(ge=0, le=2)
    tempo_espera_min: int = Field(ge=0, le=720)
    sangramento: int = Field(ge=0, le=1)
    consciencia_alterada: int = Field(ge=0, le=1)
    mobilidade_reduzida: int = Field(ge=0, le=1)

class TriagemResponse(BaseModel):
    risco: Literal["alto", "baixo", "medio"]
    probabilidades: dict[str, float]
    modelo: str
    versao_modelo: str