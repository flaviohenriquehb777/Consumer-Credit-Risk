# Fase 4 — Modelagem (Versão Corrigida: Padrão Sênior SEM LEAKAGE)

Data: 01/09/2026.
Ambiente: `uv` (Python 3.11). Seed global = 42.
Base: 150.000 clientes → split estratificado 75/25 = **112.500 (TREINO)** / **37.500 (HOLDOUT)**.

---

## 1. Contrato metodológico (resolvido o leakage da primeira versão)

| Dado | Quando é permitido ler para calcular métricas? |
|---|---|
| **TREINO** (X_train, y_train) | **Tudo o que é decisório:** split interno, CV 5-fold, tuning de hparams, seleção de modelo, threshold ótimo. |
| **HOLDOUT** (X_test, y_test) | **UMA ÚNICA VEZ, no passo final (batismo de avaliação).** Não pode ser usado para escolher hparams, threshold, modelo nem técnica de balancing. |
| Pipeline SimpleImputer / top-caps / mediana | Aprendido SOMENTE nos k-1 folds internos a cada iteração. |

Todos os artefatos foram regenerados: pickles de modelos, thresholds e CSVs de tuning. Nenhum dos resultados das primeiras rodadas da Fase 4 influenciou as decisões de hparams desta execução — o tuning foi executado do zero, olhando apenas os scores de CV no treino.

---

## 2. Hiperparâmetros padrão

- `scale_pos_weight` (XGB / LGB) = `NEG / POS = 104.980 / 7.520 = 13,96` (valor exato vindo da distribuição do **TREINO**).
- `class_weight` no RandomForest: 3 candidatos por profundidade.
- `min_samples_leaf = 50` (DecisionTree), `min_samples_leaf = 30` (RandomForest) para regularização.
- Função de custo unificada usada em TODA a avaliação: **C = 10 · N_FN + 1 · N_FP** (FN = custo 10× FP, regra do negócio).

---

## 3. Resultado do tuning SÓ por CV 5-fold NO TREINO

### 3.1 DecisionTree (grid 15 depths = 1..15)

| max_depth (vencedor) | CV ROC AUC média (± std) | CV Custo 10:1 somado 5-folds |
|---|---|---|
| **7** | **0,84982 ± 0,00377** | 63.536 |
| 8 | 0,84904 ± 0,00452 | 63.384 |
| 9 | 0,84798 ± 0,00349 | 63.065 |
| 6 | 0,84678 ± 0,00422 | 62.803 |
| 10 | 0,84422 ± 0,00308 | 62.954 |

- **Escolha:** **max_depth = 7** (melhor média de ROC AUC no treino OOF).
- Arquivo completo: `reports/evaluations/fase04_cvtune_decisiontree_*.csv`.

### 3.2 RandomForest (grid 18 = 6 depths × 3 class_weights)

Top 5 por CV ROC AUC:

| max_depth | class_weight | CV ROC AUC média (± std) |
|---|---|---|
| **12** | **balanced_1_14** | **0,86254 ± 0,00366** |
| 12 | balanced | 0,86238 ± 0,00361 |
| 10 | balanced | 0,86230 ± 0,00361 |
| 10 | balanced_1_14 | 0,86228 ± 0,00359 |
| 12 | balanced_subs | 0,86213 ± 0,00331 |

- **Escolha:** `max_depth=12`, `class_weight=balanced_1_14` (n_estimators=300, min_samples_leaf=30).

### 3.3 XGBoost (grid 18 = 3 depths × 3 lr × 2 min_child_weight)

Top 5:

| md | lr | mcw | CV ROC AUC média (± std) |
|---|---|---|---|
| **4** | **0,03** | **80** | **0,86415 ± 0,00375** |
| 4 | 0,03 | 100 | 0,86398 ± 0,00374 |
| 5 | 0,03 | 80  | 0,86332 ± 0,00351 |
| 5 | 0,03 | 100 | 0,86327 ± 0,00355 |
| 4 | 0,05 | 80  | 0,86296 ± 0,00359 |

- **Escolha:** `max_depth=4`, `learning_rate=0.03`, `min_child_weight=80`, `n_estimators=500`, `subsample=0.9`, `colsample_bytree=0.85`, `reg_alpha=0.1`, `reg_lambda=1.0`, `scale_pos_weight=13,96`.

### 3.4 LightGBM (grid 18 = 3 md × 3 lr × 2 mcs)

| md | lr | mcs | CV ROC AUC média (± std) |
|---|---|---|---|
| **5** | **0,03** | **150** | **0,86364 ± 0,00392** |
| 5 | 0,03 | 100 | 0,86346 ± 0,00385 |
| 6 | 0,03 | 150 | 0,86231 ± 0,00402 |
| 6 | 0,03 | 100 | 0,86177 ± 0,00387 |
| 5 | 0,05 | 150 | 0,86133 ± 0,00366 |

- **Escolha:** `max_depth=5`, `learning_rate=0.03`, `min_child_samples=150`, `num_leaves=31`, `n_estimators=500`, `subsample=0.9`, `colsample_bytree=0.85`, `scale_pos_weight=13,96`.

---

## 4. Threshold ótimo para a função custo 10:1 — SÓ treino OOF

Usamos as predições agregadas de **out-of-fold do TREINO** para varrer 9.801 thresholds e escolher o que minimiza `10·FN + FP`.

NUNCA o HOLDOUT foi consultado nesta etapa.

| Modelo | Threshold ótimo (treino OOF) | Custo OOF somado nos 5 folds |
|---|---|---|
| LightGBM best | **0,543** | 37.530 |
| XGBoost best  | **0,558** | 37.556 |
| RandomForest best | **0,570** | 37.686 |
| DecisionTree best | **0,083** | 39.631 |
| Dummy (prior) | 0,500 (sem otimização) | 75.200 |

Artefato para Streamlit:
```
models/thresholds_otimos_custo_10_1.pkl
{
  "DecisionTree best": 0.083,
  "RandomForest best": 0.57,
  "XGBoost best": 0.558,
  "LightGBM best": 0.543
}
```

---

## 5. BATISMO no HOLDOUT (única vez) — Avaliação cega de produção

Esta tabela foi gerada SOMENTE após TODAS as decisões acima já estarem tomadas. Nenhum hparams ou threshold foi alterado após este passo.

Tamanho do HOLDOUT = 37.500 clientes.
**Referência (Política ATUAL da instituição = "aprova todo mundo"):**
- Custo total esperado = 25.060 unidades (2.506 FN × 10 + 0 FP).
- Nenhum calote é negado (100% de FN na prática).

| Modelo (vencedor do tuning CV) | Threshold aplicado (vem do OOF treino) | ROC AUC | PR AUC | F1 (th=0,5) | Custo 10:1 (th=0,5) | FN/FP th=0,5 | Economia vs Pol. Atual (%, th=0,5) | Custo 10:1 (threshold ÓTIMO treino) | FN/FP th ÓTIMO | Economia vs Pol. Atual (%, th ÓTIMO) |
|---|---|---|---|---|---|---|---|---|---|---|
| **🏆 XGBoost best** (md=4, lr=0.03, mcw=80) | **0,558** | **0,86956** | **0,40854** | 0,3399 | 12.628 | 566 / 6.968 | 49,61% | **12.368** | 686 / 5.508 | **50,65%** |
| LightGBM best (md=5, lr=0.03, mcs=150) | 0,543 | 0,86878 | 0,40417 | 0,3408 | 12.655 | 577 / 6.885 | 49,50% | 12.526 | 671 / 5.816 | 50,02% |
| RandomForest best (d=12, cw=balanced_1_14) | 0,570 | 0,86630 | 0,39853 | 0,3426 | 12.622 | 582 / 6.802 | 49,63% | 12.491 | 769 / 4.801 | 50,16% |
| DecisionTree best (d=7) | 0,083 | 0,85263 | 0,35767 | 0,2845 | 20.656 | 2.028 / 376 | 17,57% | 13.133 | 680 / 6.333 | 47,59% |
| Dummy (prior) | 0,500 | 0,50000 | 0,06683 | 0,0000 | 25.060 | 2.506 / 0 | 0% | 25.060 | 2.506 / 0 | 0% |

---

## 6. Veredito da Fase 4

1. **Meta técnica ROC AUC ≥ 0,85:** ✅ **atingida e superada** por 4 modelos. Topo: XGBoost = **0,86956** no HOLDOUT.
2. **Meta de negócio (menor custo que a política atual):** ✅ **atingida** por todos os modelos não-triviais. **XGBoost com threshold 0,558 entrega 50,65% de economia no custo esperado da carteira** (R$12.368 vs R$25.060 a cada 37.500 clientes).
3. **Campeão final escolhido:** **XGBoost best** — maior ROC AUC, maior economia, pickles de tamanho moderado (~770 KB), 100% compatível com SHAP TreeExplainer (explicabilidade por cliente obrigatória pelo regulador).
4. **Modelo explicável secundário:** DecisionTree best (d=7) — usado para auditabilidade rápida (regras humanas), mantido como baseline "simples" junto ao XGBoost.
5. **Threshold operacional recomendado para produção:** **0,558 (XGBoost)**.

---

## 7. Artefatos persistidos nesta fase

```
├── src/
│   ├── models/fase04_modelagem_sem_leakage.py        # Script executável sênior (SEM LEAKAGE)
│   └── models/calcular_custo_10_1.py                 # (script auxiliar da pergunta, mantido para referência)
├── models/
│   ├── 01_dummy_pipeline.pkl
│   ├── 02_decisiontree_best_pipeline.pkl             # max_depth=7
│   ├── 03_randomforest_best_pipeline.pkl             # d=12, class_weight=balanced_1_14
│   ├── 04_xgboost_pipeline.pkl                       # 🏆 CAMPEÃO (md=4, lr=0.03, mcw=80)
│   ├── 05_lightgbm_pipeline.pkl                      # md=5, lr=0.03, mcs=150
│   └── thresholds_otimos_custo_10_1.pkl              # Threshold por modelo (10·FN+FP) — vêm do OOF treino
├── reports/evaluations/
│   ├── fase04_cvtune_decisiontree_*.csv
│   ├── fase04_cvtune_randomforest_*.csv
│   ├── fase04_cvtune_xgboost_*.csv
│   ├── fase04_cvtune_lightgbm_*.csv
│   ├── fase04_cv_final_ranking_*.csv                 # Ranking família a família — CV treino
│   └── fase04_holdout_avaliacao_unica_*.csv          # ⚠️ Último CSV gerado: batismo único do HOLDOUT
└── docs/fase_03b_04_pipeline_modelagem.md            # Documentação Fases 3b + 4 (legado; complementado por este)
```
