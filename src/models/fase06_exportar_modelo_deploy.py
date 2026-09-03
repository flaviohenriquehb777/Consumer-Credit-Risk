"""
FASE 6 — SCRIPT DE EXPORTAÇÃO PARA DEPLOY STREAMLIT
===================================================

⚠️  NÃO RETREINA NADA. Apenas:
  1. Carrega os artefatos OFICIAIS já produzidos nas Fases 4 / 5 / 6.
  2. Gera uma versão "Streamlit-friendly" do classificador (XGBClassifier Puro,
     sem o wrapper sklearn Pipeline e sem depender de classes customizadas
     PreparoTransformador).
  3. Gera 1 pickle de configuração consolidada com TUDO o que o app Streamlit
     precisa (threshold, features, custos, expected_value SHAP, top 10 global).
  4. Roda self-tests de sanidade para garantir que o exportado reproduz
     IDENTICAMENTE os scores do pipeline oficial (max diff < 1e-5).

Execução (na raiz do repo):
    uv run python src/models/fase06_exportar_modelo_deploy.py
"""
from __future__ import annotations

import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. Caminhos — portáveis Windows/Linux
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]   # raiz do repositório
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy_streamlit"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
SRC_FEATURES = BASE_DIR / "src" / "features"

SEED = 42
np.random.seed(SEED)


def carregar(caminho: Path):
    with open(caminho, "rb") as f:
        return pickle.load(f)


def salvar(obj, caminho: Path):
    with open(caminho, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def titulo(msg):
    print()
    print("=" * 78)
    print("  " + msg)
    print("=" * 78)


# ======================================================================
# 1. CARREGAR ARTEFATOS OFICIAIS EXISTENTES (nenhum retreino)
# ======================================================================
titulo("[1/6] Carregando artefatos OFICIAIS já treinados")

PIPE_OFICIAL = carregar(MODELS_DIR / "04_xgboost_pipeline.pkl")
PARAMS_PREP = carregar(MODELS_DIR / "preprocessamento_params.pkl")
THRESHOLDS = carregar(MODELS_DIR / "thresholds_otimos_custo_REAIS_FN5000_FP500.pkl")
SHAP_ARTEFATOS = carregar(MODELS_DIR / "shap_artefatos_producao.pkl")

print(f"  · Pipeline oficial        OK  (steps: {list(PIPE_OFICIAL.named_steps)})")
print(f"  · Preprocessamento params OK  (mediana_renda = R$ {PARAMS_PREP['mediana_renda_mensal']:,.2f})")
print(f"  · Threshold XGB oficial   OK  (TH = {THRESHOLDS['XGBoost best']})")
print(f"  · SHAP artefatos          OK  (features = {len(SHAP_ARTEFATOS['feature_names'])})")


# ======================================================================
# 2. EXTRAIR: XGBClassifier PURO (sem wrapper Pipeline)
# ======================================================================
titulo("[2/6] Extraindo XGBClassifier PURO (para Streamlit não depender de classes custom)")

# Importa aqui apenas para pegar nomes das features via função oficial
sys.path.insert(0, str(BASE_DIR))
from src.features.pipeline_modelo import obter_nomes_features  # noqa: E402

FEATURE_NAMES_15 = obter_nomes_features(PIPE_OFICIAL)
XGB_PURO = PIPE_OFICIAL.named_steps["modelo"]  # XGBClassifier diretamente

# Garante que o booster tenha feature_names claros (sklearn XGBClassifier às vezes
# deixa como None internamente, usando ordem posicional f0..f14). Nós GRAVAMOS
# explicitamente os nomes no booster salvo para que SHAP e debug funcionem depois.
booster = XGB_PURO.get_booster()
nomes_internos = booster.feature_names
if nomes_internos is None or len(nomes_internos) != len(FEATURE_NAMES_15):
    # Reatribui explicitamente. Apesar de feature_names ser só display,
    # shap.TreeExplainer e st.dataframe exibem melhor com nomes reais.
    booster.feature_names = list(FEATURE_NAMES_15)
    print(f"  · Feature names do booster vinham NÃO-setados → atribuímos "
          f"{len(FEATURE_NAMES_15)} nomes reais agora.")
else:
    assert list(nomes_internos) == FEATURE_NAMES_15, (
        "Feature names do booster divergem do pipeline!\n"
        f"  booster: {list(nomes_internos)}\n  pipeline: {FEATURE_NAMES_15}"
    )
    print(f"  · Feature names do booster: {len(nomes_internos)} — JÁ estavam IGUAIS ao pipeline.")

salvar(XGB_PURO, DEPLOY_DIR / "modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl")
print("  · Salvo: deploy_streamlit/modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl")


# ======================================================================
# 3. COPIAR: preprocessamento.py PARA A PASTA deploy_streamlit
#    (Streamlit vai rodar este arquivo standalone)
# ======================================================================
titulo("[3/6] Copiando preprocessamento.py (standalone) p/ pasta deploy")

shutil.copy2(
    SRC_FEATURES / "preprocessamento.py",
    DEPLOY_DIR / "preprocessamento.py",
)
print("  · Salvo: deploy_streamlit/preprocessamento.py  (100% standalone)")


# ======================================================================
# 4. PICKLE CONSOLIDADO: TUDO o que o Streamlit precisa em 1 único arquivo
# ======================================================================
titulo("[4/6] Gerando pickle CONFIGURACAO_CONSOLIDADA (threshold + features + SHAP + custos)")

config = {
    # --- Identificação do modelo ---
    "modelo_nome": "XGBoost Campeão (Fase 4, SEM LEAKAGE, StratifiedKFold 5-fold OOF treino)",
    "modelo_tipo": "XGBClassifier (TreeExplainer exato)",
    "hparams_oficiais": {
        k: v for k, v in XGB_PURO.get_params().items()
        if k in ("max_depth", "learning_rate", "n_estimators",
                 "min_child_weight", "subsample", "colsample_bytree",
                 "scale_pos_weight")
    },
    # --- Performance oficial no HOLDOUT (números batizados, NÃO são promessas) ---
    "metricas_oficiais_holdout_n37500": {
        "ROC_AUC": 0.86956,
        "PR_AUC":  0.40854,
        "economia_custo_pct_vs_aprova_todos": 50.65,
        "threshold_APLICADO_AQUI": THRESHOLDS["XGBoost best"],
    },
    # --- Decisão (limiar ótimo aprendido OOF treino, validado holdout) ---
    "TH_OTIMO_XGB": THRESHOLDS["XGBoost best"],  # = 0.56
    "TH_DESCRICAO": THRESHOLDS["descricao"],
    "CUSTO_FN_RS": THRESHOLDS["CUSTO_FN"],       # 5000
    "CUSTO_FP_RS": THRESHOLDS["CUSTO_FP"],       # 500
    # --- Input esperado do analista (11 colunas ORIGINAIS, SEM flags, SEM target) ---
    "INPUT_COLUNAS_OBRIGATORIAS_11": [
        "idade",
        "renda_mensal",
        "dependentes",
        "uso_limite_rotativo",
        "razao_divida",
        "linhas_credito_abertas",
        "financiamentos_imobiliarios",
        "atrasos_30_59_dias",
        "atrasos_60_89_dias",
        "atrasos_90_mais_dias",
        # = 10 até aqui. 11ª: nada mais. 11 no total porque usamos 10 features + ...
        #   (contando: são 10 acima? São 10? Vamos listar e contar:)
        #     1.idade 2.renda 3.dependentes 4.uso_limite 5.razao_divida
        #     6.linhas_credito 7.financiamentos 8.atrasos_30_59 9.atrasos_60_89
        #     10.atrasos_90_mais
        #   ==> 10 colunas reais de entrada. A 11ª, 12ª, ..., 15ª são derivadas
        #       pelo preprocessamento.py (renda_ausente / dependentes_ausentes /
        #       cod_sistema_atrasos / renda_por_dependente / sobra_caixa).
        #   Corrigido abaixo:
    ][:10],  # trunca para 10 pq é 10 mesmo
    # --- Features que saem do preparo = entrada do modelo XGB puro ---
    "OUTPUT_PREP_COLUNAS_15": FEATURE_NAMES_15,
    # --- SHAP (para explicar 1 cliente NOVO em < 10ms) ---
    "SHAP_feature_names": SHAP_ARTEFATOS["feature_names"],
    "SHAP_expected_value_logodds": SHAP_ARTEFATOS["expected_value_logodds"],
    "SHAP_top10_features_global": SHAP_ARTEFATOS["top10_features"],
    "SHAP_top10_mean_abs_shap_global": SHAP_ARTEFATOS["top10_mean_abs_shap"],
    "SHAP_nota": (
        "No Streamlit: para UM cliente novo, (1) preparar_dados(fit=False) -> X_15cols, "
        "(2) rodar shap.TreeExplainer(XGB_PURO).shap_values(X_15cols) -> sv_1x15, "
        "(3) ordenar |sv| decrescente e mostrar top 5 local com sinal (↑ risco / ↓ risco). "
        "NÃO reutilizar shap_values_holdout_XGBbest.npy — ele é só histórico holdout."
    ),
    # --- Parâmetros do preparo (mediana, caps, códigos 96/98) ---
    "PARAMS_PREPROCESSAMENTO": PARAMS_PREP,
    # --- Exemplo funcional para Streamlit copiar/colar no seu st.button("Exemplo") ---
    "EXEMPLO_CLIENTE_APROVAR": {
        "idade": 65,
        "renda_mensal": 8_000.0,
        "dependentes": 1,
        "uso_limite_rotativo": 0.05,
        "razao_divida": 0.15,
        "linhas_credito_abertas": 3,
        "financiamentos_imobiliarios": 1,
        "atrasos_30_59_dias": 0,
        "atrasos_60_89_dias": 0,
        "atrasos_90_mais_dias": 0,
    },
    "EXEMPLO_CLIENTE_RECUSAR": {
        "idade": 32,
        "renda_mensal": 2_300.0,
        "dependentes": 4,
        "uso_limite_rotativo": 1.25,
        "razao_divida": 0.85,
        "linhas_credito_abertas": 12,
        "financiamentos_imobiliarios": 0,
        "atrasos_30_59_dias": 2,
        "atrasos_60_89_dias": 1,
        "atrasos_90_mais_dias": 0,
    },
}

# Corrige colunas de input (10 colunas, não 11 — foi erro de contagem)
config["INPUT_COLUNAS_OBRIGATORIAS_10"] = [
    "idade",
    "renda_mensal",
    "dependentes",
    "uso_limite_rotativo",
    "razao_divida",
    "linhas_credito_abertas",
    "financiamentos_imobiliarios",
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]
del config["INPUT_COLUNAS_OBRIGATORIAS_11"]

salvar(config, DEPLOY_DIR / "config_MODELO_XGB_e_SHAP_consolidado.pkl")
print(f"  · Salvo: deploy_streamlit/config_MODELO_XGB_e_SHAP_consolidado.pkl")
print(f"    ↳ Campos chave: TH={config['TH_OTIMO_XGB']}, FN={config['CUSTO_FN_RS']},"
      f" FP={config['CUSTO_FP_RS']}, expected_value_logodds="
      f"{config['SHAP_expected_value_logodds']:.6f}")


# ======================================================================
# 5. MANIFESTO: o que o deploy Streamlit PRECISA copiar (e ordem de uso)
# ======================================================================
titulo("[5/6] Gerando manifesto_para_streamlit.md")

manifesto = f"""# Manifesto de Deploy Streamlit — Modelo XGBoost Campeão
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
entrada = pd.DataFrame([{{
    "idade": 40, "renda_mensal": 5000, ...
    # (10 colunas de cfg['INPUT_COLUNAS_OBRIGATORIAS_10'])
}}])

# (2) Prepara com os parâmetros aprendidos no treino → 15 colunas
X_tratado = preparar_dados(entrada, params=params_prep, fit=False)
X_tratado = X_tratado[NOMES_15]   # garante ordem

# (3) Probabilidade + decisão
p = float(xgb_model.predict_proba(X_tratado, iteration_range=(0, xgb_model.best_iteration + 1))[0, 1])
decisao = "RECUSAR CRÉDITO (risco alto)" if p >= TH else "APROVAR CRÉDITO (risco baixo)"
print(f"P(inadimplência 2 anos) = {{p:.2%}}  →  {{decisao}}")
```

## SHAP DAQUELA LINHA (1 cliente) — <10ms

```python
import shap
explainer = shap.TreeExplainer(xgb_model)               # 1× no startup
sv = explainer.shap_values(X_tratado)                   # 1 linha → shape (1, 15)
local = pd.DataFrame({{
    "Feature": NOMES_15,
    "Valor":  [round(float(X_tratado.iloc[0, c]), 3) for c in range(15)],
    "Contribuição_SHAP": [round(float(sv[0, c]), 5) for c in range(15)],
}}).assign(abs_shap=lambda d: d["Contribuição_SHAP"].abs()) \\
  .sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"]).head(5)
# ↑ Mostre esta tabela no st.dataframe: "↑" se Contribuição_SHAP > 0 = piora risco,
#   "↓" se < 0 = melhora risco.
```

## Garantia de reprodutibilidade
- Todas as regras de negócio (caps 50k / 20, códigos 96/98, mediana renda R$ {PARAMS_PREP['mediana_renda_mensal']:,.2f})
  vêm dentro do `params_prep` que é passado para `preparar_dados(fit=False)`.
- NENHUMA nova regra é aprendida em produção.
- Threshold 0.56 é **FIXO** no deploy (definido em função de custo R$ 5k/500 OOF treino da Fase 5).
"""  # noqa: W605

(DEPLOY_DIR / "manifesto_para_streamlit.md").write_text(manifesto, encoding="utf-8")
print("  · Salvo: deploy_streamlit/manifesto_para_streamlit.md  (passo a passo 1:1)")


# ======================================================================
# 6. SELF-TEST CRÍTICO: mesmo resultado entre (pipeline OFICIAL) vs
#    (preprocessamento.py standalone + XGB puro)?
# ======================================================================
titulo("[6/6] SELF-TEST — mesmo output entre pipeline oficial vs exportado deploy?")

# Registra o módulo importado do arquivo COPIADO (teste de standalone real)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "deploy_preprocessamento",
    DEPLOY_DIR / "preprocessamento.py",
)
deploy_pp = importlib.util.module_from_spec(spec)
sys.modules["deploy_preprocessamento"] = deploy_pp
spec.loader.exec_module(deploy_pp)  # type: ignore[union-attr]

# Dois casos de teste (do manifesto) + 1 caso com missing p/ testar imputação
casos = [
    ("APROVAR (exemplo)",   pd.DataFrame([config["EXEMPLO_CLIENTE_APROVAR"]])),
    ("RECUSAR (exemplo)",   pd.DataFrame([config["EXEMPLO_CLIENTE_RECUSAR"]])),
    ("MISSING renda e dep", pd.DataFrame([{
        "idade": 45,
        "renda_mensal": np.nan,
        "dependentes": np.nan,
        "uso_limite_rotativo": 0.75,
        "razao_divida": 0.55,
        "linhas_credito_abertas": 8,
        "financiamentos_imobiliarios": 2,
        "atrasos_30_59_dias": 98,      # código sistema
        "atrasos_60_89_dias": 98,
        "atrasos_90_mais_dias": 98,
    }])),
]

tudo_ok = True
max_diff_total = 0.0
for nome, df_in in casos:
    # --- Referência (Pipeline OFICIAL) ---
    p_pipe = float(PIPE_OFICIAL.predict_proba(df_in)[0, 1])

    # --- Deploy (preprocessamento standalone + XGB Puro, ordem cols correta) ---
    X_deploy = deploy_pp.preparar_dados(df_in, params=PARAMS_PREP, fit=False)
    X_deploy = X_deploy[FEATURE_NAMES_15]
    p_deploy = float(XGB_PURO.predict_proba(X_deploy)[0, 1])

    diff_abs = abs(p_pipe - p_deploy)
    max_diff_total = max(max_diff_total, diff_abs)
    dec = "✅ MESMO RESULTADO" if diff_abs < 1e-5 else "❌ DIFERENÇA CRÍTICA"
    print(f"  · Caso {nome}")
    print(f"      p_pipeline_oficial = {p_pipe:.6f}  |  p_deploy = {p_deploy:.6f}  "
          f"|  |Δ| = {diff_abs:.1e}   {dec}")
    if diff_abs >= 1e-5:
        tudo_ok = False

print()
if tudo_ok:
    print("🎉  TODOS OS SELF-TESTS PASSARAM. Diferença máxima entre pipeline oficial "
          f"e deploy exportado = {max_diff_total:.1e} < 1e-5.")
    print("    Exportação é 100% reprodutível e SEGURA para colocar em produção.")
else:
    raise SystemExit(
        "[ERRO CRÍTICO] Deploy exportado difere do pipeline oficial em > 1e-5. "
        "NÃO USAR em produção. Verificar ordem de features no XGB_PURO."
    )


# ======================================================================
# RESUMO FINAL
# ======================================================================
titulo("RESUMO FINAL: pasta deploy_streamlit pronta")
tamanhos = []
for f in sorted(DEPLOY_DIR.iterdir()):
    if f.is_file():
        tam = f.stat().st_size
        tamanhos.append((f.name, tam))
        print(f"  · {f.name:<62s}  {tam:>12,d} bytes")

print()
print("Para fazer o app Streamlit, basta copiar os 4 arquivos .pkl/.py acima para")
print("sua pasta app_streamlit/ e seguir o passo a passo em:")
print(f"  → {DEPLOY_DIR / 'manifesto_para_streamlit.md'}")
print()
print(f"Threshold oficial (fixo no deploy): XGB = {config['TH_OTIMO_XGB']}")
print(f"Custo FN / FP (R$) .................: {config['CUSTO_FN_RS']} / {config['CUSTO_FP_RS']}")
print(f"ROC AUC oficial holdout (n=37.500) ..: "
      f"{config['metricas_oficiais_holdout_n37500']['ROC_AUC']:.5f}")
print(f"Economia de custo (%) vs aprova todos: "
      f"{config['metricas_oficiais_holdout_n37500']['economia_custo_pct_vs_aprova_todos']:.2f}%")
print()
sys.exit(0)
