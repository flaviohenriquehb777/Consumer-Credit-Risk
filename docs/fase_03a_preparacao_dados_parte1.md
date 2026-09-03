# Fase 3 — Preparação dos Dados (Parte 1)

**Projeto:** Modelo de Risco de Crédito (Aurora Crédito Digital)
**Data:** 31/08/2026
**Semente:** 42
**Entrada:** `data/raw/credito_tratado.csv` (150.000 clientes × 11 colunas)
**Função única de pipeline:** [src/features/preprocessamento.py](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/src/features/preprocessamento.py)
**Notebooks:**
  - Limpo (sem outputs): [notebooks/clean/03a_preparacao_dados_pt1.ipynb](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/notebooks/clean/03a_preparacao_dados_pt1.ipynb)
  - Executado (com outputs e histórico): [notebooks/executed/03a_preparacao_dados_pt1.ipynb](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/notebooks/executed/03a_preparacao_dados_pt1.ipynb)

---

## Arquitetura do pipeline de preparação

**Arquivo:** [src/features/preprocessamento.py](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/src/features/preprocessamento.py)

- **Função única `preparar_dados(df, params, fit, target_col, test_size, seed)`.**
- Dois modos:
  - `fit=True` (projeto): faz split 75-25, aprende parâmetros só no treino, transforma treino e teste. Retorna `(X_train, X_test, y_train, y_test, params)`.
  - `fit=False` (produção / Streamlit): recebe `params` do treino, aplica **apenas transform**, sem aprender nada novo. Retorna `X_transformado`.
- Parâmetros aprendidos no treino são **serializados em pickle** — `models/preprocessamento_params.pkl` — para uso exato no app Streamlit sem recálculo e sem data leakage.

---

## Etapa 0 — Split estratificado 75-25

**Fonte:** toda a base `credito_tratado.csv` de 150.000 linhas.

**Split (seed=42, `stratify=y`):**

| Conjunto | Nº linhas | % do total | N Inadimplentes | % Inadimplentes |
|---|---|---|---|---|
| **Treino** | **112.500** | 75,0% | **7.520** | **6,68%** |
| **Teste** | **37.500** | 25,0% | **2.506** | **6,68%** |
| Base bruta | 150.000 | 100,0% | 10.026 | 6,68% |

**Conclusão:** ✔️ estratificação perfeita — mesma taxa de 6,68% em treino e teste.

---

## Etapas 1–6 — Tratamentos (aplicados com parâmetros APRENDIDOS NO TREINO)

Todos os valores aprendidos (mediana, caps, etc.) **nunca** foram olhados no teste.

| Passo | Ação | Parâmetro do treino | Resultado em X_train |
|---|---|---|---|
| 1 | Imputar `renda_mensal` faltante com mediana do treino | `mediana_renda_mensal = 5.400,00` | Nulos em renda: 0 |
| 2 | Criar flag `renda_ausente` | — | N=22.210 (19,74% de 112.500) |
| 3 | Criar flag `dependentes_ausentes` | — | N=2.961 (2,63% de 112.500) |
| 4 | Códigos 96/98 → flag única `cod_sistema_atrasos` + top-cap 20 nas 3 colunas de atraso | `top_cap_atrasos = 20`, `codigos = (96, 98)` | Flag ativada em 198 linhas (0,176%). `max` das 3 colunas = 20 |
| 5 | Top-cap renda em R$ 50.000 | `top_cap_renda = 50.000` | `max(renda_mensal) = R$ 50.000` |
| 6.1 | `renda_por_dependente = renda_mensal / (dependentes + 1)` | — | Média R$ 4.487,89, Mediana R$ 3.375,00 |
| 6.2 | `sobra_caixa = renda_mensal * (1 - razao_divida)` | — | Média R$ 2.624,27, Mediana R$ 1.560,00, Mín R$ -33.333,00 (divida > renda) |

Gráfico das distribuições após tratamento: [reports/figures/fase03_distribuicoes_apos_tratamento.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase03_distribuicoes_apos_tratamento.png)

---

## Resumo final da base tratada (X_train)

**Dimensão:** **112.500 linhas × 15 colunas** (10 originais + 3 flags + 2 features novas).

| # | Coluna | Tipo Pandas | Qtd Nulos | Mínimo | Máximo | Média | Mediana |
|---|---|---|---|---|---|---|---|
| 1 | `idade` | int64 | 0 | 18 | 103 | 52,27 | 52 |
| 2 | `renda_mensal` | float64 | 0 | 0 | 50.000 | 6.256,21 | 5.400 |
| 3 | `dependentes` | float64 | 0 | 0 | 20 | 0,738 | 0 |
| 4 | `uso_limite_rotativo` | float64 | 0 | 0 | 2,00 | 0,324 | 0,14 |
| 5 | `razao_divida` | float64 | 0 | 0 | 2,00 | 0,680 | 0,54 |
| 6 | `linhas_credito_abertas` | int64 | 0 | 0 | 58 | 8,46 | 8 |
| 7 | `financiamentos_imobiliarios` | int64 | 0 | 0 | 54 | 1,02 | 1 |
| 8 | `atrasos_30_59_dias` | int64 | 0 | 0 | 20 | 0,28 | 0 |
| 9 | `atrasos_60_89_dias` | int64 | 0 | 0 | 20 | 0,10 | 0 |
| 10 | `atrasos_90_mais_dias` | int64 | 0 | 0 | 20 | 0,13 | 0 |
| 11 | **`renda_ausente`** (flag) | int64 | 0 | 0 | 1 | 0,197 | 0 |
| 12 | **`dependentes_ausentes`** (flag) | int64 | 0 | 0 | 1 | 0,026 | 0 |
| 13 | **`cod_sistema_atrasos`** (flag) | int64 | 0 | 0 | 1 | 0,002 | 0 |
| 14 | **`renda_por_dependente`** (nova) | float64 | 0 | 0 | 50.000 | 4.487,89 | 3.375 |
| 15 | **`sobra_caixa`** (nova) | float64 | 0 | -33.333 | 49.998 | 2.624,27 | 1.560 |

**Qualidade da base:**
- **Zeramos todos os valores faltantes** (15 colunas, 0 nulos em cada).
- Todas as colunas de atraso estão com `max = 20` (top-cap funcionou).
- Renda está limitada em R$ 50.000 (top-cap funcionou).
- Flags e features novas todas criadas com os ranges esperados.

---

## Parâmetros fitados no treino (salvos em pickle para produção)

Arquivo: [models/preprocessamento_params.pkl](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/models/preprocessamento_params.pkl)

```python
{
  "mediana_renda_mensal": 5400.0,   # Aprendido só no treino
  "top_cap_atrasos":      20,       # Decisão negócio
  "top_cap_renda":        50000,    # Decisão negócio
  "codigos_sistema":      [96, 98], # Da Fase 2
  "colunas_atraso":       ['atrasos_30_59_dias', 'atrasos_60_89_dias', 'atrasos_90_mais_dias']
}
```

Como usar no Streamlit de produção:
```python
import pandas as pd
from src.features.preprocessamento import preparar_dados, carregar_params

params = carregar_params("models/preprocessamento_params.pkl")
X = preparar_dados(nova_linha_cliente, params=params, fit=False)
predicao   = modelo.predict_proba(X)[0,1]    # score 0-100%
```

---

## Artefatos salvos em disco

| Caminho | Tamanho (aprox.) | Finalidade |
|---|---|---|
| `data/processed/X_train.csv` | ~8,3 MB | Features para treinar algoritmos (Fase 4). |
| `data/processed/X_test.csv`  | ~2,8 MB | Holdout final (só usar na Fase 5). |
| `data/processed/y_train.csv` | ~340 KB | Vetor alvo de treino. |
| `data/processed/y_test.csv`  | ~115 KB | Vetor alvo de teste. |
| `models/preprocessamento_params.pkl` | ~1 KB | Parâmetros do pipeline (mediana, caps). **Leitura obrigatória no Streamlit.** |

---

## Tratamentos SUGERIDOS para a Parte 2 da Fase 3

Todos abaixo carecem de validação com número real. Apenas são hipóteses a testar.

| # | Tratamento proposto | Por que considerar |
|---|---|---|
| 1 | **Remover as 1.573 duplicatas exatas** restantes antes do split e revalidar performance | Pode reduzir overfitting artificial de árvores que aprendem "cópia exata". |
| 2 | **Top-capping em `financiamentos_imobiliarios` e `linhas_credito_abertas`** (P99 vs. 54 e 58) | Sem tratamento, os splits profundos de árvore podem ser guiados por outliers raros. |
| 3 | **Top-capping em `idade`** (P99=87 vs. max 103) | Perfil idoso raro; reduzir influência em modelos lineares. |
| 4 | **Features de interação:** total_atrasos = soma das 3 colunas; severidade_max = max das 3 | Combinação de atrasos pode ampliar o sinal monotônico confirmado na Fase 2. |
| 5 | **Feature `idade` agrupada em faixas etárias** (ex.: 18-25, 26-40, 41-60, 60+) | Permite explicabilidade "cliente jovem = score maior risco". |
| 6 | **Interações financeiras:** `utilizacao × atraso_total`; `sobra_caixa × idade` | Combinações alinhadas à intuição de negócio: sobra caixa baixa + histórico ruim = risco alto. |
| 7 | **Discretização/WoE-binning para `uso_limite_rotativo`, `razao_divida`, `renda_mensal`, `idade`** | Modelos lineares e exibilidade regulatória pedem justificativa "por faixa". |
| 8 | **StandardScaler / RobustScaler em pipeline por modelo** | Regressão Logística e modelos de distância requerem padronização; árvores não. |
| 9 | **Avaliar se caps 50k (renda) e 20 (atrasos) são thresholds ótimos via IV/Gain** (grid fino). | Valores foram heurísticas de negócio; validação numérica deixa defensável. |
| 10 | **Tratamento da sobra_caixa negativa** — pode ser cravada em 0, ou manter como está (carrega sinal de "despesas > renda"). | Atualmente -33.333 mínimo — precisa confirmar via teste se a negatividade ajuda o modelo ou se clipar em 0 melhora performance. |

---

**Próxima etapa:** Parte 2 da Fase 3 (validar os tratamentos acima com teste numérico e consolidar a base analítica final para modelagem).

Aguardando o próximo passo.
