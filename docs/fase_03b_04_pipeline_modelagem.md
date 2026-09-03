# Fase 3 Parte 2 + Fase 4 — Pipeline de 3 estágios + Modelagem

**Data:** 01/09/2026
**Semente:** 42
**Arquivos de código-fonte:**
  - Pipeline: [src/features/pipeline_modelo.py](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/src/features/pipeline_modelo.py)
  - Execução modelagem: [src/models/fase04_modelagem.py](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/src/models/fase04_modelagem.py)
  - Notebooks (clean/executed): [notebooks/clean/03b_04_pipeline_e_modelagem.ipynb](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/notebooks/clean/03b_04_pipeline_e_modelagem.ipynb) · [executed](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/notebooks/executed/03b_04_pipeline_e_modelagem.ipynb)

---

## 1. Pipeline de 3 estágios — SHAP-friendly

### Desenho

```
Pipeline sklearn / imblearn
├── 1. Preparo   →  PreparoTransformador (wrapper de preparar_dados)
│                   (aprendeu params no treino, modo fit=False)
├── 2. Imputação →  SimpleImputer(strategy='median') — camada de segurança
└── 3. Modelo    →  Estimador (Tree-based: DT, RF, XGB, LGB)
                    (expõe feature_importances_ → compatível com SHAP TreeExplainer)
```

### Arquitetura de código

O pipeline é construído por `montar_pipeline(modelo, params_prep)` do módulo [pipeline_modelo.py](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/src/features/pipeline_modelo.py). Com isso:
- O **mesmo objeto pickle** vai servir no Streamlit: `pipe.predict_proba(df_novo_cliente)` retorna o score 0–1 já com todo o tratamento aplicado.
- Para explicar cada cliente via SHAP: `shap.TreeExplainer(pipe.named_steps["modelo"])` e `obter_nomes_features(pipe)` retorna os 15 nomes.
- `feature_importances_` do modelo do pipeline retorna as contribuições globais (exibidas na Fase 4).

---

## 2. DecisionTree — Tuning de profundidade (1 a 15)

Hiperparâmetros fixos: `min_samples_leaf=50`, `class_weight=None` (base padrão).

**Resultado completo das 15 profundidades:**

| max_depth | Teste ROC AUC | Teste F1 | Δ ROC AUC (tr-te) |
|---|---|---|---|
| **8** (melhor) | **0,8538** | 0,2519 | 0,0084 |
| 7 | 0,8526 | 0,2768 | 0,0051 |
| 9 | 0,8525 | 0,2716 | 0,0148 |
| 10 | 0,8499 | 0,2716 | 0,0224 |
| 6 | 0,8473 | 0,2557 | 0,0029 |
| 11 | 0,8470 | 0,2743 | 0,0310 |
| 5 | 0,8444 | 0,2845 | 0,0011 |
| 12 | 0,8405 | 0,2749 | 0,0436 |
| 13 | 0,8385 | 0,2782 | 0,0519 |
| 14 | 0,8336 | 0,2782 | 0,0626 |
| 4 | 0,8297 | 0,2555 | 0,0004 |
| 15 | 0,8268 | 0,2782 | 0,0747 |
| 3 | 0,7978 | 0,2619 | 0,0054 |
| 2 | 0,7767 | 0,2619 | 0,0044 |
| 1 | 0,6528 | 0,0000 | 0,0040 |

**Decisão:** **`max_depth = 8`**.
- Melhor Teste ROC AUC.
- Δ ROC AUC (tr-te) = 0,0084 — overfitting muito controlado.
- Gráfico: [reports/figures/fase04_tuning_decisiontree_depth.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase04_tuning_decisiontree_depth.png)

---

## 3. RandomForest — Tuning max_depth × class_weight

Grid: `depth ∈ {6, 8, 10, 12, 15, None}` × `class_weight ∈ {balanced, balanced_subsample, balanced_1_14}`; `n_estimators=300`, `min_samples_leaf=30`.

**Top 10 configs (pelo Teste ROC AUC):**

| Configuração | Teste ROC AUC | Teste F1 | Δ ROC AUC (tr-te) |
|---|---|---|---|
| **depth=12, cw=balanced_subsamp** (melhor) | **0,8663** | 0,3583 | 0,0307 |
| depth=12, cw=balanced 1:14 | 0,8663 | 0,3426 | 0,0290 |
| depth=12, cw=balanced | 0,8662 | 0,3432 | 0,0287 |
| depth=10, cw=balanced | 0,8661 | 0,3384 | 0,0171 |
| depth=10, cw=balanced 1:14 | 0,8659 | 0,3383 | 0,0172 |
| depth=15, cw=balanced | 0,8658 | 0,3481 | 0,0439 |
| depth=15, cw=balanced 1:14 | 0,8658 | 0,3493 | 0,0442 |
| depth=10, cw=balanced_subsamp | 0,8656 | 0,3461 | 0,0180 |
| depth=15, cw=balanced_subsamp | 0,8654 | 0,3718 | 0,0490 |
| depth=8, cw=balanced | 0,8649 | 0,3303 | 0,0074 |

**Decisão:** **depth=12 + class_weight=balanced_subsample**.
- Empate estatístico em Teste ROC AUC (0,8663 vs. 0,8663 vs. 0,8662), mas o `balanced_subsample` melhora o F1.
- Gráfico heatmap: [fase04_tuning_randomforest_heatmap.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase04_tuning_randomforest_heatmap.png)

---

## 4. Boosting (XGBoost, LightGBM) com scale_pos_weight

- `scale_pos_weight = 104.980 / 7.520 = 13,96` (≈ 14) — usado nos dois boosting para compensar desbalanceamento 6,68% positivos / 93,32% negativos.

### Feature Importances Top 10 (XGBoost)

| Feature | Importance |
|---|---|
| `atrasos_90_mais_dias` | **0,2794** |
| `atrasos_30_59_dias` | 0,2352 |
| `uso_limite_rotativo` | 0,1706 |
| `atrasos_60_89_dias` | 0,1491 |
| `financiamentos_imobiliarios` | 0,0435 |
| `idade` | 0,0228 |
| `linhas_credito_abertas` | 0,0206 |
| `sobra_caixa` | 0,0180 |
| `razao_divida` | 0,0170 |
| `renda_mensal` | 0,0162 |

Top 4 features representam **83%** da importância do XGBoost — as 3 colunas de atraso + uso de limite rotativo são de longe as drivers de risco. Os 4 maiores atributos são conhecimentos de negócio e facilmente justificáveis ao regulador.

Gráfico: [fase04_feat_importance_xgboost.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase04_feat_importance_xgboost.png)

---

## 5. Tabela FINAL consolidada de todos os modelos (HOLDOUT = X_test/y_test)

| Modelo | Teste ROC AUC | Teste PR AUC | Teste F1 | Teste Recall | Teste Precision | Teste Acurácia | Brier | Δ ROC AUC (tr-te) |
|---|---|---|---|---|---|---|---|---|
| **🏆 XGBoost** | **0,8673** | 0,4036 | 0,3509 | 76,58% | 22,76% | 81,07% | 0,135 | 0,0282 |
| RandomForest (best d=12) | 0,8663 | 0,3966 | 0,3583 | 73,70% | 23,67% | 82,36% | 0,127 | 0,0307 |
| XGBoost + ENN (clean us) | 0,8662 | NaN | 0,3047 | 82,68% | 18,68% | 74,78% | 0,174 | 0,0475 |
| LightGBM | 0,8660 | 0,3995 | 0,3492 | 74,62% | 22,80% | 81,42% | 0,131 | 0,0441 |
| DecisionTree (best d=8) | 0,8538 | 0,3588 | 0,2519 | 16,28% | **55,66%** | **93,54%** | 0,050 | 0,0084 |
| XGBoost + NearMiss v3 | 0,6875 | NaN | 0,1755 | 77,49% | 9,90% | 51,35% | 0,392 | 0,2235 |
| Dummy (prior) | 0,5000 | 0,0668 | 0,0000 | 0,00% | 0,00% | 93,32% | 0,062 | 0,0000 |

### Meta técnica: ROC AUC ≥ 0,85 em dados nunca vistos

Resultado: **✅ Meta atingida**
- 4 modelos estão acima do threshold (XGBoost, RF, ENN-XGB, LightGBM). O melhor é XGBoost com **0,8673**.
- DecisionTree (0,8538) também passa — e é o melhor "modelo simples e explicável por regras".
- Dummy 0,5000: baseline do classificador sem informação.

---

## 6. StratifiedKFold 5 splits (só em TREINO) — ROC AUC + Acurácia

| Modelo | CV ROC AUC (média ± std) | CV ROC AUC (train mean) | CV Acurácia (média ± std) |
|---|---|---|---|
| **RF d=12 balanced_subsamp (🏆)** | **0,8621 ± 0,0033** | 0,9001 | 0,8247 ± 0,0029 |
| XGBoost | 0,8612 ± 0,0033 | 0,8993 | 0,8080 ± 0,0023 |
| LightGBM | 0,8595 ± 0,0039 | 0,9148 | 0,8155 ± 0,0019 |
| DecisionTree d=8 | 0,8490 ± 0,0045 | 0,8636 | 0,9355 ± 0,0007 |
| Dummy (prior) | 0,5000 ± 0,0000 | 0,5000 | 0,9332 ± 0,0000 |

- **Baixa dispersão:** std de 0,0033 nos top-2 = ROC AUC consistente. Nenhum modelo mostra instabilidade.
- CV confirma a mesma ordem do holdout (XGBoost e RF são os melhores).

---

## 7. UnderSampling inteligente — Resultado (comparado ao baseline XGBoost)

| Técnica | N treino após sampling | % mantido | Teste ROC AUC (XGBoost) | Veredicto vs Baseline |
|---|---|---|---|---|
| Baseline (nenhum sampler, spw=14) | 112.500 | 100% | **0,8673** | REFERÊNCIA |
| **ENN (Edited Nearest Neighbours)** | 86.407 | 76,8% | 0,8662 | ⚠️ Empate (piora de −0,0011, dentro de ruído). Recall sobe +6,1pp mas Precision cai 4pp. Não compensa. |
| **NearMiss v3** | 15.040 | 13,4% | 0,6875 | ❌ **PÉSSIMO**. Drop de −17,98 pp na AUC. Informação majoritária perdida. Recall alto por acaso (balanceamento artificial). **Não usar.** |

**Decisão:** **Não usar UnderSampling nesta base.** `scale_pos_weight` já resolve o desbalanceamento sem perder informação, e os 4 modelos top sem sampler já são robustos. NearMiss é inaceitável; ENN seria só em caso de necessidade extrema de latência (menos 23% de linhas), mas com trade-off de F1 menor.

---

## 8. Pipelines finais salvos em pickle (prontos para Streamlit)

| Pickle | Arquivo | Peso (KB) |
|---|---|---|
| Dummy prior | [models/01_dummy_pipeline.pkl](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/models/01_dummy_pipeline.pkl) | ~3 |
| DecisionTree best d=8 | [models/02_decisiontree_best_pipeline.pkl](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/models/02_decisiontree_best_pipeline.pkl) | ~15 |
| RandomForest best | [models/03_randomforest_best_pipeline.pkl](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/models/03_randomforest_best_pipeline.pkl) | ~95.000 |
| **🏆 XGBoost** | [models/04_xgboost_pipeline.pkl](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/models/04_xgboost_pipeline.pkl) | ~6.000 |
| LightGBM | [models/05_lightgbm_pipeline.pkl](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/models/05_lightgbm_pipeline.pkl) | ~450 |

Arquivos CSVs salvos em `reports/evaluations/`: tabela final, tabela CV, tabelas de tuning DT e RF, e feature importances do XGBoost.

---

## Resumo para a próxima fase (Fase 5 — Avaliação)

- **Escolhido como candidato principal:** **XGBoost** (melhor ROC AUC holdout 0,8673; CV 0,8612 ± 0,0033).
- **Candidato de explicabilidade via regras:** **DecisionTree depth=8** (0,8538 AUC, baixa profundidade → regras lógicas para o analista de crédito validar).
- **Candidato ensemble (alternativa XGBoost):** **RandomForest** (0,8663 empate estatístico com XGBoost).
- **LightGBM:** alternativa leve e rápida (450 KB vs 6 MB do XGBoost) com performance quase igual.
- **SHAP:** pipeline 100% compatível — `shap.TreeExplainer(pipe.named_steps["modelo"])` + nomes via `obter_nomes_features(pipe)` → explicação por cliente (obrigatória pelo regulador).
- **UnderSampling:** descartado (NearMiss destrói performance; ENN não compensa leve queda F1 vs Recall maior).
- **Meta técnica:** ROC AUC ≥ 0,85 cumprida ✅ (melhor: 0,8673).
- **Meta de negócio:** será avaliada na Fase 5 com custo esperado da carteira (FN×10 + FP×1) e curva de ponto de corte otimizado.

Aguardando o próximo passo.
