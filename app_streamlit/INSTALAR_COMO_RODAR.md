# 🏦  COMO RODAR O APP DE RISCO DE CRÉDITO (XGBoost Campeão)

> Pasta onde está tudo: **`app_streamlit/`** (compartilhe ESSA pasta com o time).

---

## 1. O que tem dentro da pasta `app_streamlit/`?

Todos os arquivos **PRECISAM estar na MESMA pasta**. Não mova nem renomeie nada.

| Arquivo | O que é |
|---|---|
| `app_risco_credito.py` | **O app do Streamlit propriamente dito (1 arquivo único).** |
| `requirements.txt` | Lista de bibliotecas + versões PINNADAS. |
| `modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl` | Modelo XGB treinado e otimizado. NÃO RETREINAR. |
| `preprocessamento.py` | Regras de negócio do preparo (flags, caps 50k / 20, imputação mediana, features novas). |
| `config_MODELO_XGB_e_SHAP_consolidado.pkl` | Configuração consolidada: TH=0,56, FN=R$5.000, FP=R$500, top 10 global SHAP, 2 exemplos de clientes. |

---

## 2. Instalação — 2 opções (qualquer uma serve)

### Opção A — usando PIP (funciona no 99% dos PCs Windows/Mac/Linux, mais fácil)

**Abra o Terminal (PowerShell no Windows, Terminal no Mac/Linux)** e execute:

```powershell
# Entra na pasta do app (IMPORTANTE)
cd caminho\para\a\pasta\app_streamlit

# Cria um ambiente virtual (recomendado)
python -m venv .venv

# Ativa o ambiente virtual:
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   Windows CMD:         .venv\Scripts\activate.bat
#   Mac / Linux:         source .venv/bin/activate
.venv\Scripts\Activate.ps1

# Instala TODAS as bibliotecas (streamlit + xgboost + shap + pandas + sklearn + numpy)
pip install --upgrade pip
pip install -r requirements.txt
```

### Opção B — usando UV (mais rápido, usado durante o projeto)

```powershell
cd caminho\para\a\pasta\app_streamlit
uv venv
uv pip install -r requirements.txt
```

> Resumo do que o requirements.txt instala (6 libs):
> `streamlit`, `xgboost`, `scikit-learn`, `shap`, `pandas`, `numpy`.

---

## 3. Abrir o app

Com o ambiente virtual **ativado** e dentro da pasta `app_streamlit/`:

```powershell
streamlit run app_risco_credito.py
```

Isto abrirá automaticamente o navegador em `http://localhost:8501/`.

---

## 4. Como usar o app (3 cliques)

1. **Preencha** os 10 campos do formulário, ou use os botões **"Exemplo: APROVAR"** / **"Exemplo: RECUSAR"** para carregar casos de teste.
2. Clique no botão verde **"🔍 Avaliar risco deste cliente"**.
3. Leia os 3 resultados:
   - **Probabilidade de inadimplência** (%);
   - **Decisão** — `APROVAR CRÉDITO` se p < 0,56, `RECUSAR CRÉDITO` se p ≥ 0,56;
   - **TOP 5 SHAP local** — as 5 features que MAIS influenciaram ESSA decisão daquele cliente (↑ aumenta risco / ↓ diminui risco).

**Dica:** sempre confira a coluna "Consistência SHAP" (terceiro card); se aparecer "OK ✅", a previsão e a explicação batem perfeitamente.

---

## 5. Compartilhamento com a equipe — SIM, funciona perfeitamente

**É SIM perfeitamente possível compartilhar com a equipe e ele funcionar para todos.** Basta enviar a **pasta `app_streamlit/` INTEIRA** (por zip, Teams, OneDrive, repo Git compartilhado etc.) e cada pessoa da equipe segue o passo **2. Instalação** em seu computador. Nenhum arquivo adicional é necessário.

Se quiser evitar que cada membro instale localmente (melhor ainda), existem 3 opções de deploy 1 clique com o mesmo código do app — você escolhe posteriormente quando quiser:

| Plataforma | Dificuldade | Como |
|---|---|---|
| **Streamlit Community Cloud** (gratuita) | Muito fácil | Crie uma conta em `https://streamlit.io/cloud`, aponte para o seu repo GitHub, defina `app_streamlit/app_risco_credito.py` como entrypoint. Pronto. Link público p/ time. |
| **Servidor interno (Windows Server / Linux)** | Média | `pip install streamlit` → `nssm` agendar como serviço Windows. Dá URL interna da empresa. |
| **Docker** (qualquer cloud AWS/Azure/GCP) | Média | `python:3.11-slim` → `pip install -r requirements.txt` → `EXPOSE 8501` → `CMD streamlit run ...`. 1 Dockerfile de 8 linhas. |

O código do app (`app_risco_credito.py`) NÃO precisa de nenhuma alteração para ir para qualquer um desses 3 ambientes.

---

## 6. FAQ rápida

| Pergunta | Resposta |
|---|---|
| Preciso do Python instalado? | Sim, **Python 3.10 ou 3.11** (3.11 recomendado — mesma versão usada no projeto). |
| O modelo retreina? | **NÃO.** O classificador .pkl foi treinado com a base de 150k clientes e nunca mais retreina. |
| Threshold 0,56 pode ser alterado em produção? | Sim, basta editar `TH_OTIMO_XGB` dentro do `config_MODELO_XGB_e_SHAP_consolidado.pkl` (ou no próprio app, antes do deploy). |
| E se eu receber código 96/98 nos campos de atraso do sistema de origem? | **Informe exatamente assim (não normalize).** O `preprocessamento.py` detecta e cria a flag binária `cod_sistema_atrasos` automaticamente. |
| Missing de renda? | **Deixe 0 (ou NaN).** O modelo flagga como `renda_ausente=1` e imputa mediana R$ 5.400,00 aprendida no treino. O analista NÃO precisa imputar nada. |
| App demora para carregar? | Primeira carga ~2–4s (carrega XGBoost e compila o TreeExplainer). Cada avaliação de cliente nova é **< 120 ms** (na maioria dos casos < 20 ms). |
| Onde foi treinado? | Repositório: `Consumer-Credit-Risk`. Fase 4 tuning StratifiedKFold 5-fold (só OOF treino). Fase 6 exportação. O arquivo `manifesto_para_streamlit.md` documenta. |

---

## 7. Quem pode usar?

Público-alvo do app (conforme escopo inicial do projeto):
  1. **Analista de crédito** (usuário principal: preenche e recebe decisão + justificativa p/ cliente);
  2. **Área de Risco** (altera limiares, acompanha top 10 global e validação);
  3. **Regulador / cliente final** (coluna TOP 5 SHAP fornece a explicação individual exigida por LGPD / BCB).

**Para o analista de crédito usar no dia a dia:** só precisa preencher os 10 campos e apertar um botão. Nenhum conhecimento de Python/Machine Learning é necessário.

Bom uso.
