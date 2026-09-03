# Manifesto de Deploy Streamlit — Modelo XGBoost Campeão
> Gerado por `src/models/fase06_exportar_modelo_deploy.py`. **NÃO retreinamos nada aqui.**

## Arquivos obrigatórios na pasta `app_streamlit/`

Copie estes 4 arquivos para a pasta do seu app Streamlit:

| Arquivo | O que é | Tipo |
|---|---|---|
| `modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl` | XGBClassifier treinado e já otimizado. **SEM sklearn Pipeline wrapper.** | Pickle |
| `preprocessamento.py` | Rotina de preparação 100% standalone (flags + caps + imputação + novas features). Não requer nenhum outro arquivo de `src/`. | Código Python |
| `config_MODELO_XGB_e_SHAP_consolidado.pkl` | 1 único dict com: TH_OTIMO_XGB=0.56, CUSTO_FN_RS=5000, CUSTO_FP_RS=500, nomes das 10 cols de INPUT, 15 cols de OUTPUT do preparo, SHAP expected_value e top10 global, 2 exemplos de clientes (1 aprovar, 1 recusar). | Pickle |

## Ordem de USO no Streamlit (3 passos)

```python
import pickle, pandas as pd
from preprocessamento import preparar_dados, carregar_params  # do arquivo copiado

# (0) Carregar tudo 1 vez no startup
with open("modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl","rb") as f:
    xgb_model = pickle.load(f)
with open("config_MODELO_XGB_e_SHAP_consolidado.pkl","rb") as f:
    cfg = pickle.load(f)
params_prep = cfg["PARAMS_PREPROCESSAMENTO"]
TH           = cfg["TH_OTIMO_XGB"]          # 0.56
NOMES_15     = cfg["OUTPUT_PREP_COLUNAS_15"]

# (1) Analista preenche 10 campos → DataFrame 1 linha
entrada = pd.DataFrame([{
    "idade": 40, "renda_mensal": 5000, ...
    # (10 colunas de cfg['INPUT_COLUNAS_OBRIGATORIAS_10'])
}])

# (2) Prepara com os parâmetros aprendidos no treino → 15 colunas
X_tratado = preparar_dados(entrada, params=params_prep, fit=False)
X_tratado = X_tratado[NOMES_15]   # garante ordem

# (3) Probabilidade + decisão
p = float(xgb_model.predict_proba(X_tratado, iteration_range=(0, xgb_model.best_iteration + 1))[0, 1])
decisao = "RECUSAR CRÉDITO (risco alto)" if p >= TH else "APROVAR CRÉDITO (risco baixo)"
print(f"P(inadimplência 2 anos) = {p:.2%}  →  {decisao}")
```

## SHAP DAQUELA LINHA (1 cliente) — <10ms

```python
import shap
explainer = shap.TreeExplainer(xgb_model)               # 1× no startup
sv = explainer.shap_values(X_tratado)                   # 1 linha → shape (1, 15)
local = pd.DataFrame({
    "Feature": NOMES_15,
    "Valor":  [round(float(X_tratado.iloc[0, c]), 3) for c in range(15)],
    "Contribuição_SHAP": [round(float(sv[0, c]), 5) for c in range(15)],
}).assign(abs_shap=lambda d: d["Contribuição_SHAP"].abs()) \
  .sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"]).head(5)
# ↑ Mostre esta tabela no st.dataframe: "↑" se Contribuição_SHAP > 0 = piora risco,
#   "↓" se < 0 = melhora risco.
```

## Garantia de reprodutibilidade
- Todas as regras de negócio (caps 50k / 20, códigos 96/98, mediana renda R$ 5,400.00)
  vêm dentro do `params_prep` que é passado para `preparar_dados(fit=False)`.
- NENHUMA nova regra é aprendida em produção.
- Threshold 0.56 é **FIXO** no deploy (definido em função de custo R$ 5k/500 OOF treino da Fase 5).
