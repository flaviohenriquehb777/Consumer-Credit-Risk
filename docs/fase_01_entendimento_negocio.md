# Fase 1 — Entendimento do Negócio

**Projeto:** Modelo de Risco de Crédito para Previsão de Inadimplência  
**Empresa:** Instituição Financeira (carteira de crédito pessoal)  
**Data:** 31/08/2026  
**Semente:** 42  

---

## 1. Contexto Operacional Atual

A instituição recebe pedidos de crédito e, hoje, a decisão de aprovar ou negar cada proposta é tomada por meio de:
- Regras manuais predefinidas.
- Experiência e julgamento subjetivo do analista de crédito.

Esse processo apresenta duas fontes principais de ineficiência:
1. **Inconsistência:** dois analistas podem decidir de forma diferente para o mesmo perfil.
2. **Baixa escalabilidade:** a velocidade e a qualidade da decisão dependem da disponibilidade humana.

O objetivo do projeto é **apoiar** essa decisão com um modelo preditivo, não substituí-la totalmente.

---

## 2. Objetivo do Modelo

Estimar a **probabilidade de calote** (inadimplência) para cada pedido de crédito, com base em dados históricos de clientes.  
A probabilidade é utilizada como insumo para três públicos distintos:

| Grupo de Usuários | Como Usa o Resultado do Modelo |
| --- | --- |
| **Analista de Crédito** | Recebe o score na tela e decide aprovar ou negar a proposta. |
| **Área de Risco** | Define a política de aprovação — onde fica a linha de corte do score. |
| **Regulador e Cliente** | Exigem saber **por quê** o crédito foi negado (explicabilidade obrigatória). |

---

## 3. Definição do Alvo

- **Coluna alvo:** `inadimplente_2anos`
- **Classe positiva (1):** cliente ficou 90 dias ou mais em atraso (90+ DPD) em qualquer momento nos 2 anos seguintes à concessão do crédito.
- **Classe negativa (0):** cliente não atingiu 90 dias de atraso no período observado.

Essa definição segue a prática padrão do mercado financeiro brasileiro para caracterização de inadimplência material.

---

## 4. Estrutura de Erros e Matriz de Custos

### Dois tipos de erro possíveis

| Tipo de Erro | Nome Técnico | Descrição de Negócio | Impacto |
| --- | --- | --- | --- |
| Aprovar quem vai dar calote | **Falso Negativo (FN)** | Cliente ruim entra na carteira, gera perda de principal. | **10x mais caro** |
| Negar quem teria pago | **Falso Positivo (FP)** | Cliente bom é recusado, perde-se margem e participação de mercado. | Custo de oportunidade |

### Razão de custos definida pelo negócio

**Custo(FN) = 10 × Custo(FP)**

Essa razão é o pilar que irá guiar:
- A escolha da métrica principal (não basta acurácia bruta).
- A determinação do ponto de corte ótimo na política de aprovação.
- A avaliação econômica final do modelo na carteira.

---

## 5. Critérios de Sucesso do Projeto

### 5.1 Critério Técnico
- **ROC AUC ≥ 0,85** em dados de teste (holdout nunca visto pelo modelo durante treinamento).
- Justificativa: ROC AUC mede a capacidade de ordenamento do modelo (distinguir bons de pagadores de maus pagadores), independentemente do ponto de corte — adequado para problema com custos desiguais.

### 5.2 Critério de Negócio
- **Custo esperado da carteira sob o modelo < custo esperado da política atual (regras manuais + experiência do analista).**
- O custo esperado é calculado pela fórmula:

```
Custo Esperado = (Número de FNs × Custo_FN) + (Número de FPs × Custo_FP)
```

### 5.3 Critério Operacional (não negociável)
- **Explicabilidade:** toda decisão negada deve ter uma justificativa compreensível por humano (analista, cliente, regulador). Serão utilizadas técnicas como SHAP / LIME além de análise de importância de variáveis do próprio modelo.

---

## 6. Restrições e Premissas

### Restrições
1. **Idioma:** código comentado em português; nomes de variáveis e colunas em português.
2. **Reprodutibilidade:** toda operação com aleatoriedade utilizará `SEED = 42`.
3. **Decisões baseadas em evidência:** nenhuma recomendação de técnica sem o número, tabela ou gráfico correspondente.
4. **Avaliação de custo:** o ponto de corte final não será escolhido por máxima acurácia, mas por mínimo custo esperado da carteira.

### Premissas Iniciais (a validar na Fase 2 — Entendimento dos Dados)
1. O dataset `credito_tratado.csv` contém registros históricos completos o suficiente para modelagem.
2. As features disponíveis (demográficas, financeiras, histórico de atraso) têm poder preditivo relevante.
3. Não há vazamento temporal no dataset (informações de pós-concessão contaminando o treino).
4. A razão de custos 10:1 será mantida durante todo o projeto — a menos que a área de negócio informe o contrário.

---

## 7. Resumo Executivo da Fase 1

> "Não precisamos de um score bonito. Precisamos de uma decisão economicamente defensável."
> — Diretora de Risco

Este projeto é, acima de tudo, um **problema de decisão sob incerteza com custos assimétricos.**  
O papel do modelo não é apenas "classificar bem" no sentido estatístico, mas **reduzir o custo total da carteira** de crédito da instituição, mantendo:
- Risco controlado (menos FNs = menos calotes).
- Volume suficiente de aprovações (menos FPs = menos clientes perdidos).
- Transparência total (explicabilidade para todos os stakeholders).

---

**Próxima fase:** Fase 2 — Entendimento dos Dados (análise exploratória dos dados em `data/raw/credito_tratado.csv`).
