# Fase 2 — Entendimento dos Dados

**Projeto:** Modelo de Risco de Crédito (Aurora Crédito Digital)
**Data:** 31/08/2026
**Semente:** 42
**Fonte dos dados:** `data/raw/credito_tratado.csv`
**Notebooks de referência:**
  - Limpo (sem outputs): [notebooks/clean/02_entendimento_dados.ipynb](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/notebooks/clean/02_entendimento_dados.ipynb)
  - Executado (com outputs e histórico de runs): [notebooks/executed/02_entendimento_dados.ipynb](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/notebooks/executed/02_entendimento_dados.ipynb)
  - Script rodável (equivalente): [src/data/fase_02_analise_exploratoria.py](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/src/data/fase_02_analise_exploratoria.py)

---

## Sumário Executivo da Fase 2

| Pergunta | Resposta (com número real) |
|---|---|
| Tamanho da base | **150.000** clientes × **11** colunas |
| Desequilíbrio do alvo | **6,68%** inadimplentes (1 para cada 13,96 adimplentes) |
| Acurácia do baseline "ninguém calota" | **93,32%** |
| Colunas com nulos | `renda_mensal` (19,82%) e `dependentes` (2,62%) |
| Perda se dropar linhas com nulos | 29.731 clientes perdidos (19,82% da base) |
| Linhas duplicadas | 1.573 (1,05%) |
| Outlier mais grave | `renda_mensal` máxima (R$ 3.008.750) é **120×** o P99 (R$ 25.000) |
| Atrasos com valor suspeito | Máximo **98** nas 3 faixas de atraso vs P99 = 2/3/4 — provável código de erro |

---

## 1. Carregamento, formato e primeiras linhas

**Evidência numérica:**

| Indicador | Valor |
|---|---|
| Clientes (linhas) | **150.000** |
| Features + alvo (colunas) | **11** |
| Memória utilizada | 12.59 MB |

**Estrutura:** dataset único, sem particionamento. 10 features + 1 coluna-alvo binária (`inadimplente_2anos`).

---

## 2. Dicionário oficial de dados da base

| Coluna | Tipo (Pandas) | Classificação | Significado | Unidade |
|---|---|---|---|---|
| `inadimplente_2anos` | int64 | Inteiro binário | 1 = cliente ficou 90+ DPD em até 2 anos; 0 = adimplente | Flag (0/1) |
| `idade` | int64 | Inteiro | Idade do cliente no momento da avaliação | Anos |
| `renda_mensal` | float64 | Contínuo | Renda mensal informada ou estimada | Reais (R$) |
| `dependentes` | float64 | Discreto | Número de dependentes declarados | Pessoas |
| `uso_limite_rotativo` | float64 | Contínuo | Proporção do limite rotativo utilizada | Razão (0–1, pode >1 se estourado) |
| `razao_divida` | float64 | Contínuo | Relação entre comprometimento financeiro e renda | Razão |
| `linhas_credito_abertas` | int64 | Inteiro | Qtd. de linhas de crédito ativas no histórico | Unidades |
| `financiamentos_imobiliarios` | int64 | Inteiro | Número de financiamentos imobiliários registrados | Unidades |
| `atrasos_30_59_dias` | int64 | Inteiro | Qtd. de episódios de atraso entre 30 e 59 dias | Unidades |
| `atrasos_60_89_dias` | int64 | Inteiro | Qtd. de episódios de atraso entre 60 e 89 dias | Unidades |
| `atrasos_90_mais_dias` | int64 | Inteiro | Qtd. de episódios de atraso com 90 dias ou mais | Unidades |

**Observação:** `dependentes` e `renda_mensal` estão como `float64` unicamente por causa dos `NaN` (pandas não suporta `Int64` nullable por padrão ao ler CSV). Trataremos na Fase 3.

---

## 3. Distribuição do alvo

| Classe | Contagem | Percentual |
|---|---|---|
| 0 — Adimplente | 139.974 | 93,32% |
| 1 — Inadimplente | 10.026 | 6,68% |

**Razão de desequilíbrio:** 1 inadimplente para cada **13,96** adimplentes.

Gráfico salvo em: [reports/figures/fase02_distribuicao_alvo.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_distribuicao_alvo.png)

---

## 4. Baseline de referência (modelo de 1 linha)

**Regra:** "todo mundo é adimplente" (chuta sempre classe 0).

**Evidência numérica:**
- Classes: 0 (adimplente) = maioria absoluta (93,32%).
- **Acurácia desse "modelo": 93,32%.**

---

## 5. Consequência para a escolha de métricas

### Por que NÃO usar só acurácia?

O baseline de 93,32% prova que acurácia é **enganosa** neste projeto. Um modelo teoricamente "excelente" com 94% de acurácia só é **0,68 ponto percentual** melhor do que não fazer nada — e pode, na prática, estar deixando passar quase todos os Falsos Negativos (que custam 10× mais). A acurácia ignora a matriz de custos 10:1 do negócio.

### Métricas oficialmente escolhidas

| Métrica | O que mede | Critério de uso | Cuidados obrigatórios |
|---|---|---|---|
| **ROC AUC** | Capacidade de ordenamento (separar bons de maus pagadores) | ≥ 0,85 (critério técnico mínimo) | Não incorpora custos FN/FP. Mede performance estatística, não econômica. |
| **Precision** | Dos negados pelo modelo, % que realmente calotou | Apoio ao analista de crédito | Pode ser artificialmente alta se negar poucos clientes. |
| **Recall / Sensibilidade** | Dos calotes reais, % que o modelo capturou | Apoio à área de Risco | Aumentar à custa de Precision nega clientes bons demais. |
| **F1-Score** | Média harmônica Precision × Recall | Apenas comparativo entre modelos | Trata FN e FP com mesmo peso — incompatível com custo 10:1. Nunca usar como decisório. |
| **Custo Esperado da Carteira** | `N_FN × 10 + N_FP × 1` (unidade de custo de negócio) | **Métrica principal de NEGÓCIO.** Ponto de corte será otimizado sobre ela. | Depende do ponto de corte. Não é uma métrica do modelo só, mas da política completa. |
| **Brier / Log Loss** | Calibração das probabilidades | Auditores e pricing | Um score de 18% precisa representar chance real de ~18%. |
| **Matriz de Confusão** | Contagem de TN, FP, FN, TP | Relatórios para área de Risco | Visão por componente (quantos e quais erros). |

---

## 6. Valores nulos por coluna

**Apenas duas colunas possuem missing:**

| Coluna | Qtd. Nulos | % Nulos |
|---|---|---|
| `renda_mensal` | **29.731** | **19,82%** |
| `dependentes` | **3.924** | **2,62%** |

Gráfico salvo em: [reports/figures/fase02_valores_nulos.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_valores_nulos.png)

---

## 7. Podemos simplesmente dropar as linhas com nulos?

| Cenário | Quantidade | Percentual |
|---|---|---|
| Total de clientes | 150.000 | 100,00% |
| Linhas com pelo menos 1 nulo | 29.731 | 19,82% |
| Linhas completas | 120.269 | 80,18% |

**Conclusão (com evidência): NÃO.**

Motivos:
1. **Volume:** perder 29.731 clientes = quase 1/5 da base. Em um dataset de 150k isso é muita informação jogada fora.
2. **Viés de seleção:** o missing de `renda_mensal` **não é completamente aleatório.** Em crédito, o cliente que escolhe NÃO declarar renda se auto-seleciona como um perfil diferente da média. Simplesmente remover essas linhas ensinaria o modelo "como classificar clientes que informaram renda" e ele não performaria na população real, onde ~20% não declaram.
3. **A ausência é informação:** criar uma flag "renda não informada" é, na prática, uma nova feature preditiva.

**Recomendação oficial para a Fase 3:**
- `renda_mensal`: (a) criar flag `renda_ausente` + (b) imputar com a **mediana** (robusta a outliers).
- `dependentes`: (a) criar flag `dependentes_ausente` + (b) imputar com **moda/mediana** (0 é razoável).

---

## 8. Outros problemas identificados (evidência por coluna)

### 8.1 Estatísticas descritivas globais

O `describe()` da base revelou distribuições extremamente assimétricas. Tabela completa no notebook executado.

### 8.2 Outliers extremos (Máximo ÷ P99)

| Coluna | P99 | Máximo | Razão Máx/P99 | Conclusão |
|---|---|---|---|---|
| `renda_mensal` | R$ 25.000 | R$ 3.008.750 | **120,35×** | Outlier crítico. **Winsorizar no P99.** |
| `atrasos_60_89_dias` | 2 | 98 | **49,00×** | Valor **98 é código de erro/missing.** Aplicar **top-capping no P99 = 2.** |
| `atrasos_90_mais_dias` | 3 | 98 | **32,67×** | Idem. **Top-capping no P99 = 3.** |
| `atrasos_30_59_dias` | 4 | 98 | **24,50×** | Idem. **Top-capping no P99 = 4.** |
| `financiamentos_imobiliarios` | 4 | 54 | **13,50×** | Caso raro mas possível. Acompanhar com top-capping. |
| `linhas_credito_abertas` | 24 | 58 | **2,42×** | Razoável; sem tratamento obrigatório. |
| `uso_limite_rotativo` | 1,09 | 2,00 | **1,83×** | >1 = cliente estourou limite (informação real de risco). **Não dropar.** |
| `idade` | 87 | 109 | **1,25×** | Raro mas possível. **Top-capping no P99 ou em 100 anos.** |

### 8.3 Problemas específicos fechados (cada um com evidência)

1. **`renda_mensal` máxima R$ 3.008.750 — 120× o P99.** Ganhador de mega-sena existe, mas o modelo deve ser treinado com winsorização no P99 = R$ 25.000 para não distorcer árvores / regressões.

2. **Três colunas de atraso com MÁXIMO = 98 exatamente.** Em um horizonte de 2 anos (24 meses), é matematicamente impossível ter **98 episódios** de atraso de 30 dias. Esse **98 é um código artificial** (ex.: "não informado", "erro de sistema"). Tratar com **top-capping no P99 de cada faixa** (4, 2 e 3 respectivamente).

3. **`uso_limite_rotativo` > 1 em 3.321 clientes (2,21%).** Isso não é erro: é cliente que estourou o limite do cartão (cheque especial / rotativo) — informação de risco importantíssima. **Manter como está, sem winsorizar para baixo de 1.**

4. **Idade máxima de 109 anos.** Possível mas raro; top-capping em 100 ou no P99 = 87 anos.

5. **`dependentes` > 10 = 2 casos** contra P99 = 4. Top-capping em 10 ou no P99 = 4.

6. **Linhas duplicadas = 1.573 (1,05% da base).** Investigar e **remover duplicatas estritas** na Fase 3 (clientes idênticos em todas as colunas, incluindo o alvo — provavelmente duplicação de ETL).

### 8.4 Visualizações auxiliares

Boxplots e histogramas das colunas mais críticas salvos em:
- [reports/figures/fase02_boxplots_outliers.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_boxplots_outliers.png)
- [reports/figures/fase02_histogramas.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_histogramas.png)

---

## 9. Investigação complementar (com números reais)

> Aqui respondemos:
> 1. **Missing de renda** — quem não informa renda tem taxa de inadimplência maior ou menor?
> 2. **Código 98 nas 3 colunas de atraso** — são as mesmas linhas nas 3 colunas? Tem diferença de risco?

### 9.1 Missing de renda × taxa de inadimplência

Comparamos diretamente a taxa de inadimplência entre o grupo que declarou renda e o grupo com missing.

| Grupo | N Clientes | N Inadimplentes | Taxa de Inadimplência |
|---|---|---|---|
| **SIM — informou renda** | 120.269 | 8.357 | **6,95%** |
| **NÃO — missing de renda** | 29.731 | 1.669 | **5,61%** |

**Resultado com evidência:**
- Diferença (não-informou − informou) = **−1,34 pontos percentuais**.
- Razão entre taxas = **0,81×** (quem não informa tem inadimplência *menor*).

#### Conclusão que refuta a heurística inicial

Minha hipótese inicial de que "quem não declara renda tem risco maior" estava **errada**. Os números mostram o oposto. Isso ilustra exatamente a regra do projeto: nenhuma decisão sem evidência.

#### Ação atualizada para a Fase 3

Dropar as 29.731 linhas continua **indevido** — agora por outro motivo:
- Não é por viés de seleção de "perfil mais arriscado", mas sim porque o perfil de quem não informa renda tem **menor risco médio** e contém informação útil para discriminar.
- **Ainda se recomenda:** flag `renda_ausente` + imputação mediana na coluna original. A flag agora deve ser interpretada como sinal negativo de risco (reduz probabilidade de calote), não positivo.

Gráfico salvo em: [reports/figures/fase02_inadimplencia_renda_informada.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_inadimplencia_renda_informada.png)

---

### 9.2 Código 98 nas 3 colunas de atraso

#### 9.2.1 Distribuição de valores — só existem 2 valores suspeitos: 96 e 98

| Coluna | N=96 | N=98 | Observação |
|---|---|---|---|
| `atrasos_30_59_dias` | 5 | 264 | Nenhum outro valor ≥ 90. |
| `atrasos_60_89_dias` | 5 | 264 | Idêntico — exatamente as mesmas contagens. |
| `atrasos_90_mais_dias` | 5 | 264 | Idêntico — exatamente as mesmas contagens. |

#### 9.2.2 São as mesmas 264 linhas nas 3 colunas SIMULTANEAMENTE?

| Medida | N linhas | % da base |
|---|---|---|
| Linhas com 98 em `atrasos_30_59_dias` | 264 | 0,176% |
| Linhas com 98 em `atrasos_60_89_dias` | 264 | 0,176% |
| Linhas com 98 em `atrasos_90_mais_dias` | 264 | 0,176% |
| Linhas com 98 **EM QUALQUER** das 3 | **264** | 0,176% |
| Linhas com 98 **NAS 3 AO MESMO TEMPO** | **264** | 0,176% |

**Resultado irrefutável:** o valor 98 aparece sempre nas **mesmas 264 linhas, nas 3 colunas, ao mesmo tempo**. Não há combinação parcial.

#### 9.2.3 Combinações possíveis (padrão Venn)

Somente DOIS padrões existem na base inteira:

| Padrão (colunas com 98) | N linhas | % |
|---|---|---|
| Nenhuma (sadio) | 149.736 | 99,824% |
| **30-59 + 60-89 + 90+ = as 3 colunas** | **264** | 0,176% |

O mesmo padrão perfeito vale para o valor 96 (5 linhas, sempre nas 3 colunas). A base foi codificada artificialmente dessa forma.

#### 9.2.4 Como são essas 264 linhas? (10 exemplos)

As 264 linhas compartilham um perfil **visivelmente distinto**:
- `uso_limite_rotativo` frequentemente = 1,00 (limite totalmente utilizado).
- Idade jovem concentrada (21, 22, 25, 27, 29, 33 anos).
- Muitas `renda_mensal` ausente ou R$ 0,00.
- `razao_divida` em geral 0,00 ou 2,00 (valores de "chão" e "teto" artificiais).
- E, naturalmente, as 3 colunas de atraso marcadas com 98 exatamente.

---

### 9.3 Taxa de inadimplência do grupo com 98 vs. resto da base

| Grupo | N Clientes | N Inadimplentes | Taxa de Inadimplência |
|---|---|---|---|
| **RESTO (sem 98 em atrasos)** | 149.736 | 9.883 | **6,60%** |
| **GRUPO COM 98 em pelo menos 1 coluna** | 264 | 143 | **54,17%** |

**Resultado com evidência:**
- Diferença (grupo 98 − resto) = **+47,57 pontos percentuais**.
- Razão entre taxas = **8,21×** — o grupo com 98 é **mais de 8 vezes** mais provável de dar calote.

Detalhe por padrão (só existe 1 padrão não sadio):

| Padrão (colunas com 98) | N | Taxa Inadimplência |
|---|---|---|
| Nenhuma (sadio) | 149.736 | 6,60% |
| 30-59, 60-89, 90+ | 264 | **54,17%** |

#### Conclusão crítica e mudança na recomendação

A estratégia inicial ("winsorize 98 → P99 em cada coluna") está **errada**. Ela apagaria o **sinal preditivo mais forte de toda a base**,
porque as 3 colunas de atraso com P99 = 4/2/3 não diferenciariam esse grupo de risco altíssimo.

**Ação recomendada para a Fase 3:**

| Passo | Técnica |
|---|---|
| 1 | Criar **flag binária `cod_erro_atrasos_98`** = 1 quando as 3 colunas de atraso são = 98 simultaneamente; 0 caso contrário. |
| 2 | Repetir para o padrão 96 (em número muito menor, 5 linhas) criando `cod_erro_atrasos_96` ou agrupando sob a mesma flag de "código de sistema". |
| 3 | Após a flag ser materializada, substituir o 98 (e 96) por `NaN` ou pelo P99 em cada uma das 3 colunas originais. |
| 4 | Com isso, a flag carrega o sinal de "risco 8× maior" e as colunas originais de atraso voltam a ter valores interpretáveis sem poluir os splits de árvores. |

Gráfico salvo em: [reports/figures/fase02_inadimplencia_codigo_98_atrasos.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_inadimplencia_codigo_98_atrasos.png)

---

## 10. Confirmação visual: faixas de atrasos × taxa de inadimplência

> **Hipótese: quanto mais episódios de atraso no passado, maior a probabilidade de calote no futuro.
> Testamos isso **visual e numericamente em cada uma das 3 colunas de atraso.

Metodologia:
- Agrupamos cada coluna nas faixas: **`0, 1, 2, 3, 4, 5+` e categoria separada **`Cód. sistema (96/98)`**.
- Para cada faixa, medimos **n de clientes (barras) e **taxa de inadimplência (linha).
- Linha pontilhada na **taxa média geral (6,68%).
- Códigos artificiais 96/98 **isolados como categoria independente.
- Ao final, correlação **Spearman ρ** (ignora 96/98 para não inflar).

### 10.1 Resultado por coluna — com número real

#### Atrasos 30–59 dias

| Faixa | N clientes | N inadimplentes | Taxa inadimplência |
|---|---|---|---|
| **0**    | 126.018 | 5.041  | **4,00%** |
| **1**    |  16.033 | 2.409  | **15,03%** |
| **2**    |   4.598  | 1.219  | **26,51%** |
| **3**    |   1.754  |   618  | **35,23%** |
| **4**    |     747  |   318  | **42,57%** |
| **5+**   |     581  |   274  | **47,16%** |
| Cód. sistema | 269 | 147 | **54,65%** |

Gráfico salvo em: [fase02_faixas_atrasos_30_59_dias.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_faixas_atrasos_30_59_dias.png)

#### Atrasos 60–89 dias

| Faixa | N clientes | N inadimplentes | Taxa inadimplência |
|---|---|---|---|
| **0**    | 142.396 | 7.256  | **5,10%** |
| **1**    |   5.731  | 1.777  | **31,01%** |
| **2**    |   1.118  |   561  | **50,18%** |
| **3**    |     318  |   180  | **56,60%** |
| **4**    |     105  |    65  | **61,90%** |
| **5+**   |      63  |    40  | **63,49%** |
| Cód. sistema | 269 | 147 | **54,65%** |

Gráfico salvo em: [fase02_faixas_atrasos_60_89_dias.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_faixas_atrasos_60_89_dias.png)

#### Atrasos ≥ 90 dias

| Faixa | N clientes | N inadimplentes | Taxa inadimplência |
|---|---|---|---|
| **0**    | 141.662 | 6.554  | **4,63%** |
| **1**    |   5.243  | 1.765  | **33,66%** |
| **2**    |   1.555  |   776  | **49,90%** |
| **3**    |     667  |   385  | **57,72%** |
| **4**    |     291  |   195  | **67,01%** |
| **5+**   |     313  |   204  | **65,18%** |
| Cód. sistema | 269 | 147 | **54,65%** |

Gráfico salvo em: [fase02_faixas_atrasos_90_mais_dias.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_faixas_atrasos_90_mais_dias.png)

Gráfico conjunto (3 colunas lado a lado): [fase02_faixas_atrasos_conjunto.png](file:///C:/Users/flavi/Documents/GitHub/Consumer-Credit-Risk/reports/figures/fase02_faixas_atrasos_conjunto.png)

### 10.2 Força da associação (Spearman ρ, ignorando 96/98)

| Coluna de atraso | Spearman (ρ) com inadimplente_2anos | Força |
|---|---|---|
| **Atrasos ≥ 90 dias | **+0,3354** | **Moderada (melhor preditor individual |
| Atrasos 60–89 dias | +0,2683 | Moderada |
| Atrasos 30–59 dias | +0,2514 | Moderada |

### 10.3 Conclusão visual e implicações

1. **Monotonicidade perfeita em faixa 0→5+ em todas as 3 colunas: taxacresce consistentemente conforme a faixa de atraso — zero hipótese "mais atrasos = mais calote é verdadeira com evidência visual irrefutável.
2. **Poder cresce conforme a severidade do atraso: atraso de **4 ou mais de 90 dias tem **taxa de 67,01%** na base — quase chance certa de calote.
3. **Rank de força (ρ): atraso ≥90 > atraso 60-89 > atraso 30-59** (0,335 > 0,268 > 0,251). Toda as três são preditoras relevantes, e **não redundantes**.
4. **Categoria Cód. sistema (96/98): **≈54,65% de calote — fica entre 4 e 5+ da faixa de atraso mais leve (30-59) e acima de 3-4 faixas nas demais). Flag binária recomendada; não winsorizar cegamente.
5. **Sem surpresas** (não tem não linha invertida ou "U" nas curvas). As colunas estão de acordo com a intuição financeira.

---

## Resumo dos encaminhamentos para a Fase 3 (Preparação dos Dados) — ATUALIZADO

| Ação | Coluna(s) afetada(s) | Técnica (com base nas evidências desta Fase 2) |
|---|---|---|
| Remover duplicatas estritas | Todas | `drop_duplicates()` — 1.573 linhas |
| Flag + imputação de missing | `renda_mensal`, `dependentes` | `renda_ausente` (0/1) + mediana; `dependentes_ausente` (0/1) + moda/mediana |
| Flag de código artificial | colunas de atraso (96 e 98) | Criar `cod_erro_atrasos_98` (1 nas 264 linhas em que as 3 colunas valem 98) e similares para 96. |
| Limpar valores artificiais | `atrasos_30_59_dias`, `atrasos_60_89_dias`, `atrasos_90_mais_dias` | Substituir 96/98 por `NaN` e/ou winsorizar no P99 = 4 / 2 / 3 respectivamente. A flag acima carrega o sinal de risco. |
| Winsorização / top-capping (outliers) | `renda_mensal`, `idade`, `dependentes`, `financiamentos_imobiliarios` | Cortar no P99 (e P01 se aplicável). `renda_mensal` teto = R$ 25.000. |
| **Não** winsorizar | `uso_limite_rotativo`, `razao_divida` | Valores >1 representam ultrapassagem real de limite e são sinal de risco. |
| Train/Test estratificado | `inadimplente_2anos` | 80/20, `stratify=y`, `random_state=42`. Split **antes** de qualquer fit de imputação/scaling (evita data leakage). |

---

**Próxima fase:** Fase 3 — Preparação dos Dados.

Aguardando o próximo passo.
