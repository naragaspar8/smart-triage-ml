from fastapi import FastAPI

from predictor import METADATA, prever_risco
from schemas import TriagemRequest, TriagemResponse

app = FastAPI ()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model": METADATA["model_name"],
        "model_version": METADATA["model_version"]
    }

@app.post("/predict",
          response_model=TriagemResponse,
)
def predict(triagem: TriagemRequest):
    dados = triagem.model_dump()

    resultado = prever_risco(dados)

    return resultado