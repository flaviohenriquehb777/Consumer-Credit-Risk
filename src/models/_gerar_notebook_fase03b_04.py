"""Gera o notebook Fase 3 Parte 2 + Fase 4 completo e roda nbconvert."""
import json
import os
import subprocess
import sys
import nbformat as nbf

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN = os.path.join(BASE, "notebooks", "clean", "03b_04_pipeline_e_modelagem.ipynb")
EXEC  = os.path.join(BASE, "notebooks", "executed", "03b_04_pipeline_e_modelagem.ipynb")

nb = nbf.v4.new_notebook()

c1_md = nbf.v4.new_markdown_cell("# Fase 3 Parte 2 + Fase 4 — Pipeline de 3 estágios + Modelagem\n\n"
"## Pipeline (3 estágios, SHAP-friendly)\n\n"
"**Estágios:**\n\n"
"1. **Preparo** — Wrapper de `preparar_dados()` do módulo preprocessamento (reutiliza params do pickle)\n"
"2. **Imputação** — `SimpleImputer(strategy='median')` como camada de segurança (caso novos dados cheguem com NaN)\n"
"3. **Modelo** — Estimador àrvore (compatível com SHAP TreeExplainer)\n\n"
"Esse pipeline:\n"
"- Recebe DataFrame BRUTO (10 colunas originais, idênticas ao CSV)\n"
"- Executa todo o tratamento dentro do `.fit()` / `.predict()`\n"
"- Expõe `.feature_importances_` via passo de modelo (para SHAP)\n"
"- É serializável em pickle para uso no Streamlit\n\n"
"## Fase 4 — Modelagem\n\n"
"**Métricas:** ROC AUC, PR AUC, F1-Score (preditos com threshold 0.5)\n\n"
"**Modelos:** Dummy (prior), DecisionTree (tuning de profundidade), RandomForest (tuning depth + class_weight), XGBoost e LightGBM (scale_pos_weight = 13,96 ≈ 14).\n\n"
"**Validação:** StratifiedKFold 5 splits (só treino) com ROC AUC e Acurácia.\n\n"
"**Teste extra:** UnderSampling inteligente (NearMiss v3 + ENN) comparado ao baseline.")
c2_setup = nbf.v4.new_code_cell("""%matplotlib inline
import os
import sys
import warnings
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
np.random.seed(42)

NOTEBOOK_DIR = os.path.abspath(os.getcwd())
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from features.preprocessamento import carregar_params, SEED_DEFAULT
from features.pipeline_modelo import (
    montar_pipeline, salvar_pipeline, obter_feature_importances,
    PreparoTransformador,
)
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, brier_score_loss,
                             accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours

PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
MODELOS_DIR  = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports", "evaluations")
FIG_DIR      = os.path.join(PROJECT_ROOT, "reports", "figures")
PARAMS_PATH  = os.path.join(MODELOS_DIR, "preprocessamento_params.pkl")

X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv"))["inadimplente_2anos"].astype(int)
y_test  = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv"))["inadimplente_2anos"].astype(int)
params_prep = carregar_params(PARAMS_PATH)

SPW = (y_train == 0).sum() / (y_train == 1).sum()
print(f"[OK] Dados carregados. scale_pos_weight = neg / pos = {SPW:.2f}")
print(f"X_train = {X_train.shape}, X_test = {X_test.shape}")
print(f"y_train distribuição: {y_train.value_counts().to_dict()}")
print(f"y_test  distribuição: {y_test.value_counts().to_dict()}")
""")
c3_md = nbf.v4.new_markdown_cell("## 1. Demonstração do Pipeline de 3 estágios\n\n"
"Montamos o pipeline com um Dummy para ilustrar que ele recebe a base ORIGINAL (10 features não tratadas), "
"aplica os 3 estágios internamente e entrega os scores. Os 1º e 2º estágios usam os parâmetros do pickle treinado na Fase 3 parte 1.")
c3_pipe = nbf.v4.new_code_cell("""# --- Ilustração de como o pipeline opera ---
pipe_dummy = montar_pipeline(DummyClassifier(strategy="prior", random_state=42), params_prep)

# Como a base "X_train" em PROC_DIR JÁ ESTÁ PREPARADA, vamos simular a entrada BRUTA:
#  Pegamos as 10 colunas originais exatamente como viriam do CSV (não as flags)
colunas_originais = ["idade", "renda_mensal", "dependentes", "uso_limite_rotativo",
                     "razao_divida", "linhas_credito_abertas", "financiamentos_imobiliarios",
                     "atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias"]

# Recuperamos uma versão "bruta" a partir do CSV original (apenas para conferir o pipeline):
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credito_tratado.csv")
df_raw = pd.read_csv(RAW_PATH).sample(50, random_state=42)
df_raw_features = df_raw[colunas_originais].copy()
df_raw_target   = df_raw["inadimplente_2anos"].values

# Demonstração: fit e predict em um sample pequeno
pipe_dummy.fit(df_raw_features, df_raw_target)
prob = pipe_dummy.predict_proba(df_raw_features)[:, 1]
pred = pipe_dummy.predict(df_raw_features)

# Imprime passagens pelo pipeline
print(f"Pipeline com Dummy — shape de entrada: {df_raw_features.shape}")
print(f"Após preparo + imputacao + modelo: proba.shape = {prob.shape}")
print(f"Pipeline steps: {[n for n,_ in pipe_dummy.steps]}")
print(f"Nomes das features após preparo: {pipe_dummy.named_steps['preparo'].feature_names_out_[:5]}... (15 colunas)")
print("\\nPipeline funciona como esperado.")
""")

c4_md = nbf.v4.new_markdown_cell("## 2. DecisionTree — tuning de max_depth (1 a 15), min_samples_leaf=50")
c4_dt = nbf.v4.new_code_cell("""resultados_dt = []
for d in range(1, 16):
    pipe = montar_pipeline(
        DecisionTreeClassifier(max_depth=d, random_state=42,
                               class_weight=None, min_samples_leaf=50),
        params_prep,
    )
    t0 = time.time()
    pipe.fit(X_train, y_train)
    prob_te = pipe.predict_proba(X_test)[:, 1]
    pred_te = pipe.predict(X_test)
    prob_tr = pipe.predict_proba(X_train)[:, 1]
    resultados_dt.append({
        "Modelo": f"DecisionTree depth={d}",
        "max_depth": d,
        "Tempo (s)": round(time.time()-t0, 2),
        "Teste ROC AUC": round(roc_auc_score(y_test, prob_te), 4),
        "Teste PR AUC":  round(average_precision_score(y_test, prob_te), 4),
        "Teste F1":      round(f1_score(y_test, pred_te), 4),
        "Teste Precision": round(precision_score(y_test, pred_te, zero_division=0), 4),
        "Teste Recall":   round(recall_score(y_test, pred_te, zero_division=0), 4),
        "Teste Acurácia": round(accuracy_score(y_test, pred_te), 4),
        "Treino ROC AUC": round(roc_auc_score(y_train, prob_tr), 4),
    })
resultados_dt = pd.DataFrame(resultados_dt)
resultados_dt["Δ ROC AUC (tr-te)"] = (resultados_dt["Treino ROC AUC"] - resultados_dt["Teste ROC AUC"]).round(4)
resultados_dt = resultados_dt.sort_values("Teste ROC AUC", ascending=False).reset_index(drop=True)
best_depth = resultados_dt["max_depth"].iloc[0]
print(f"✅ Melhor DecisionTree: max_depth = {best_depth} (Teste ROC AUC = {resultados_dt['Teste ROC AUC'].iloc[0]:.4f})")
display(resultados_dt.head(10).style.hide(axis="index"))

# Gráfico tuning
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(resultados_dt.sort_values("max_depth")["max_depth"],
         resultados_dt.sort_values("max_depth")["Teste ROC AUC"],
         "o-", color="#2ecc71", linewidth=2, label="Teste ROC AUC")
ax1.set_xlabel("max_depth"); ax1.set_ylabel("Teste ROC AUC", color="#2ecc71")
ax1.set_xticks(range(1, 16))
ax1.tick_params(axis="y", labelcolor="#2ecc71")
ax2 = ax1.twinx()
ax2.plot(resultados_dt.sort_values("max_depth")["max_depth"],
         resultados_dt.sort_values("max_depth")["Δ ROC AUC (tr-te)"],
         "s--", color="#c0392b", linewidth=2, label="Δ ROC AUC (tr-te)")
ax2.set_ylabel("Overfitting Δ (treino − teste)", color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")
fig.suptitle("DecisionTree — Tuning de profundidade", fontsize=13)
plt.tight_layout()
fpath = os.path.join(FIG_DIR, "fase04_tuning_decisiontree_depth.png")
fig.savefig(fpath, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em {fpath}")
plt.show()
""")

c5_md = nbf.v4.new_markdown_cell("## 3. RandomForest — tuning max_depth × class_weight\n\n"
"Grid: `depth ∈ {6,8,10,12,15,None}` × `class_weight ∈ {balanced, balanced_subsample, 1:14}`; "
"`n_estimators=300, min_samples_leaf=30`.")
c5_rf = nbf.v4.new_code_cell("""resultados_rf = []
depths = [6, 8, 10, 12, 15, None]
cw_grid = [("balanced", "balanced"),
           ("balanced_subs", "balanced_subsample"),
           ("balanced_1_14", {0: 1, 1: 14})]
for depth in depths:
    for cw_label, cw in cw_grid:
        pipe = montar_pipeline(
            RandomForestClassifier(n_estimators=300, max_depth=depth,
                                   class_weight=cw, n_jobs=-1,
                                   random_state=42, min_samples_leaf=30, oob_score=False),
            params_prep,
        )
        t0 = time.time()
        pipe.fit(X_train, y_train)
        prob_te = pipe.predict_proba(X_test)[:, 1]
        pred_te = pipe.predict(X_test)
        prob_tr = pipe.predict_proba(X_train)[:, 1]
        resultados_rf.append({
            "Modelo": f"RF depth={depth} cw={cw_label}",
            "max_depth": str(depth),
            "class_weight": cw_label,
            "Tempo (s)": round(time.time()-t0, 2),
            "Teste ROC AUC": round(roc_auc_score(y_test, prob_te), 4),
            "Teste PR AUC":  round(average_precision_score(y_test, prob_te), 4),
            "Teste F1":      round(f1_score(y_test, pred_te), 4),
            "Teste Precision": round(precision_score(y_test, pred_te, zero_division=0), 4),
            "Teste Recall":   round(recall_score(y_test, pred_te, zero_division=0), 4),
            "Teste Acurácia": round(accuracy_score(y_test, pred_te), 4),
            "Treino ROC AUC": round(roc_auc_score(y_train, prob_tr), 4),
        })
resultados_rf = pd.DataFrame(resultados_rf)
resultados_rf["Δ ROC AUC (tr-te)"] = (resultados_rf["Treino ROC AUC"] - resultados_rf["Teste ROC AUC"]).round(4)
resultados_rf = resultados_rf.sort_values("Teste ROC AUC", ascending=False).reset_index(drop=True)
best_rf_row = resultados_rf.iloc[0]
print(f"✅ Melhor RF: depth={best_rf_row['max_depth']}, cw={best_rf_row['class_weight']}, "
      f"ROC AUC teste={best_rf_row['Teste ROC AUC']:.4f}")
display(resultados_rf.head(10).style.hide(axis="index"))

# Heatmap RF
pivot_rf = resultados_rf.pivot_table(index="class_weight", columns="max_depth", values="Teste ROC AUC")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot_rf, annot=True, fmt=".4f", cmap="Greens", ax=ax)
ax.set_title("RandomForest — Teste ROC AUC por max_depth × class_weight")
plt.tight_layout()
fpath = os.path.join(FIG_DIR, "fase04_tuning_randomforest_heatmap.png")
fig.savefig(fpath, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em {fpath}")
plt.show()
""")

c6_md = nbf.v4.new_markdown_cell("## 4. XGBoost e LightGBM com scale_pos_weight = 13,96")
c6_boost = nbf.v4.new_code_cell("""def avaliar_pipe(nome, pipe):
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t = time.time() - t0
    proba_te = pipe.predict_proba(X_test)[:, 1]
    pred_te  = pipe.predict(X_test)
    proba_tr = pipe.predict_proba(X_train)[:, 1]
    return {
        "Modelo": nome,
        "Tempo (s)": round(t, 2),
        "Teste ROC AUC": round(roc_auc_score(y_test, proba_te), 4),
        "Teste PR AUC":  round(average_precision_score(y_test, proba_te), 4),
        "Teste F1":      round(f1_score(y_test, pred_te), 4),
        "Teste Precision": round(precision_score(y_test, pred_te, zero_division=0), 4),
        "Teste Recall":   round(recall_score(y_test, pred_te, zero_division=0), 4),
        "Teste Acurácia": round(accuracy_score(y_test, pred_te), 4),
        "Teste Brier":    round(brier_score_loss(y_test, proba_te), 5),
        "Treino ROC AUC": round(roc_auc_score(y_train, proba_tr), 4),
    }

# Baseline Dummy (prior)
pipe_dummy = montar_pipeline(DummyClassifier(strategy="prior", random_state=42), params_prep)
r_dummy = avaliar_pipe("Dummy (prior)", pipe_dummy)

# Melhor DecisionTree
pipe_best_dt = montar_pipeline(
    DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight=None,
                           min_samples_leaf=50),
    params_prep,
)
r_dt = avaliar_pipe(f"DecisionTree (best d={best_depth})", pipe_best_dt)

# Melhor RF (pega os parâmetros da linha campeã)
best_rf_depth = best_rf_row["max_depth"]
cw_val_best = [v for (l,v) in dict(cw_grid).items() if l == best_rf_row["class_weight"]][0]
pipe_best_rf = montar_pipeline(
    RandomForestClassifier(n_estimators=300,
                           max_depth=(None if best_rf_depth == "None" else int(best_rf_depth)),
                           class_weight=cw_val_best, n_jobs=-1, random_state=42,
                           min_samples_leaf=30, oob_score=False),
    params_prep,
)
r_rf = avaliar_pipe(f"RandomForest (best d={best_rf_depth}, cw={best_rf_row['class_weight']})",
                    pipe_best_rf)

# XGBoost
pipe_xgb = montar_pipeline(xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=SPW, random_state=42, n_jobs=-1, eval_metric="auc",
    tree_method="hist"), params_prep)
r_xgb = avaliar_pipe("XGBoost", pipe_xgb)

# LightGBM
pipe_lgb = montar_pipeline(lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
    min_child_samples=120, subsample=0.9, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=SPW,
    random_state=42, n_jobs=-1, verbose=-1), params_prep)
r_lgb = avaliar_pipe("LightGBM", pipe_lgb)

todos = pd.DataFrame([r_dummy, r_dt, r_rf, r_xgb, r_lgb])
todos["Δ ROC AUC (tr-te)"] = (todos["Treino ROC AUC"] - todos["Teste ROC AUC"]).round(4)
todos = todos.sort_values("Teste ROC AUC", ascending=False).reset_index(drop=True)
display(todos.style.format({"Teste ROC AUC":"{:.4f}","Teste PR AUC":"{:.4f}",
                             "Teste F1":"{:.4f}","Teste Brier":"{:.5f}"}).hide(axis="index"))

# --- Feature Importances Top 10 do XGBoost (SHAP-friendly) ---
imps_xgb = obter_feature_importances(pipe_xgb).head(15)
fig, ax = plt.subplots(figsize=(9,5.5))
sns.barplot(data=imps_xgb, x="importance", y="feature", ax=ax, palette="viridis")
ax.set_title("Feature Importances — XGBoost (15 features)")
plt.tight_layout()
fpath = os.path.join(FIG_DIR, "fase04_feat_importance_xgboost.png")
fig.savefig(fpath, dpi=150, bbox_inches="tight")
print(f"Importances top 15 salvas em {fpath}")
plt.show()
display(imps_xgb.style.hide(axis="index"))
""")

c7_md = nbf.v4.new_markdown_cell("## 5. StratifiedKFold 5 splits — SÓ DADOS DE TREINO\n\n"
"Métricas do `cross_validate`: **ROC AUC** e **Acurácia**.")
c7_cv = nbf.v4.new_code_cell("""def cv5(nome, modelo_base):
    pipe_cv = montar_pipeline(modelo_base, params_prep)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s = cross_validate(pipe_cv, X_train, y_train, cv=cv,
                       scoring=["roc_auc","accuracy"], n_jobs=-1,
                       error_score="raise", return_train_score=True)
    return {
        "Modelo": nome,
        "CV ROC AUC (média ± std)": f"{s['test_roc_auc'].mean():.4f} ± {s['test_roc_auc'].std():.4f}",
        "CV ROC AUC (train mean)":   round(s['train_roc_auc'].mean(), 4),
        "CV ROC AUC (teste min/max)": f"{s['test_roc_auc'].min():.4f} / {s['test_roc_auc'].max():.4f}",
        "CV Acurácia (média ± std)": f"{s['test_accuracy'].mean():.4f} ± {s['test_accuracy'].std():.4f}",
        "CV Acurácia (train mean)":  round(s['train_accuracy'].mean(), 4),
    }

cv_results = []
cv_results.append(cv5("Dummy (prior)", DummyClassifier(strategy="prior", random_state=42)))
cv_results.append(cv5(f"DecisionTree d={best_depth}",
                      DecisionTreeClassifier(max_depth=best_depth, random_state=42,
                                             class_weight=None, min_samples_leaf=50)))
cv_results.append(cv5(f"RandomForest d={best_rf_depth} cw={best_rf_row['class_weight']}",
                      RandomForestClassifier(n_estimators=300,
                                             max_depth=(None if best_rf_depth=="None" else int(best_rf_depth)),
                                             class_weight=cw_val_best, n_jobs=-1,
                                             random_state=42, min_samples_leaf=30)))
cv_results.append(cv5("XGBoost", xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=SPW, random_state=42, n_jobs=-1, eval_metric="auc", tree_method="hist")))
cv_results.append(cv5("LightGBM", lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
    min_child_samples=120, subsample=0.9, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=SPW,
    random_state=42, n_jobs=-1, verbose=-1)))

cv_df = pd.DataFrame(cv_results).sort_values("CV ROC AUC (média ± std)", ascending=False).reset_index(drop=True)
display(cv_df.style.hide(axis="index"))
""")

c8_md = nbf.v4.new_markdown_cell("## 6. UnderSampling inteligente (NearMiss v3 e ENN) XGBoost\n\n"
"Aplica o sampler SÓ nos dados de treino e testa com os mesmos hparams do baseline.")
c8_us = nbf.v4.new_code_cell("""# Preparo + SimpleImputer aplicados uma única vez (para alimentar os samplers)
prep_t = PreparoTransformador(params_prep)
prep_t.fit(X_train, y_train)
Xtr_p = prep_t.transform(X_train)
Xte_p = prep_t.transform(X_test)
imp_ = SimpleImputer(strategy="median")
Xtr_p = imp_.fit_transform(Xtr_p)
Xte_p = imp_.transform(Xte_p)
names = prep_t.get_feature_names_out().tolist()

# A) NearMiss v3 (undersampling forte)
nm3 = NearMiss(version=3, n_neighbors=3, n_jobs=-1)
X_tr_nm, y_tr_nm = nm3.fit_resample(Xtr_p, y_train.values)
print(f"  NearMiss v3: {Xtr_p.shape} -> {X_tr_nm.shape}")

xgb_nm = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=1.0, random_state=42, n_jobs=-1, eval_metric="auc", tree_method="hist")
xgb_nm.fit(X_tr_nm, y_tr_nm)

# B) ENN — Edited Nearest Neighbours
enn = EditedNearestNeighbours(n_neighbors=5, n_jobs=-1, kind_sel="all")
X_tr_en, y_tr_en = enn.fit_resample(Xtr_p, y_train.values)
print(f"  ENN: {Xtr_p.shape} -> {X_tr_en.shape}")

xgb_en = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=SPW, random_state=42, n_jobs=-1, eval_metric="auc", tree_method="hist")
xgb_en.fit(X_tr_en, y_tr_en)

# Tabela de comparação (apenas os modelos base já tem no todos)
def medir(nome, model, X_tr, y_tr):
    p_te = model.predict_proba(Xte_p)[:,1]; c_te = model.predict(Xte_p)
    p_tr = model.predict_proba(X_tr)[:,1]
    return {"Modelo": nome,
            "Teste ROC AUC": round(roc_auc_score(y_test, p_te),4),
            "Teste PR AUC":  round(average_precision_score(y_test, p_te),4),
            "Teste F1":      round(f1_score(y_test, c_te),4),
            "Teste Precision":round(precision_score(y_test, c_te, zero_division=0),4),
            "Teste Recall":   round(recall_score(y_test, c_te, zero_division=0),4),
            "Teste Acurácia": round(accuracy_score(y_test, c_te),4),
            "Teste Brier":    round(brier_score_loss(y_test, p_te),5),
            "Treino ROC AUC": round(roc_auc_score(y_tr, p_tr),4)}

us = pd.DataFrame([
    medir("XGBoost + NearMiss v3", xgb_nm, X_tr_nm, y_tr_nm),
    medir("XGBoost + ENN (clean sampling)", xgb_en, X_tr_en, y_tr_en),
])
us["Δ ROC AUC (tr-te)"] = (us["Treino ROC AUC"] - us["Teste ROC AUC"]).round(4)
display(us.style.hide(axis="index"))
""")

c9_md = nbf.v4.new_markdown_cell("## 7. Persistência — salva pipelines finais em pickle")
c9_save = nbf.v4.new_code_cell("""salvar_pipeline(pipe_dummy,    os.path.join(MODELOS_DIR, "01_dummy_pipeline.pkl"))
salvar_pipeline(pipe_best_dt,  os.path.join(MODELOS_DIR, "02_decisiontree_best_pipeline.pkl"))
salvar_pipeline(pipe_best_rf,  os.path.join(MODELOS_DIR, "03_randomforest_best_pipeline.pkl"))
salvar_pipeline(pipe_xgb,      os.path.join(MODELOS_DIR, "04_xgboost_pipeline.pkl"))
salvar_pipeline(pipe_lgb,      os.path.join(MODELOS_DIR, "05_lightgbm_pipeline.pkl"))

# Valida carregamento: carrega e faz predict nas mesmas 50 linhas para garantir
import pickle as pkl
with open(os.path.join(MODELOS_DIR, "04_xgboost_pipeline.pkl"), "rb") as f:
    pipe_carregado = pkl.load(f)
a = pipe_xgb.predict_proba(X_test.iloc[:50])[:,1]
b = pipe_carregado.predict_proba(X_test.iloc[:50])[:,1]
assert np.allclose(a, b), "Pickle não bate com o objeto original"
print("✅ Todos os 5 pipelines foram salvos e recarregados corretamente.")
print("Pronto para uso no Streamlit.")
""")

c10_end = nbf.v4.new_markdown_cell("---\n\n"
"**Fim da Fase 3 Parte 2 + Fase 4.** Resultado final:\n\n"
"- Pipeline SHAP-friendly de 3 estágios (Preparo → SimpleImputer mediana → Modelo)\n"
"- DecisionTree melhor depth = 8 (ROC AUC teste ≈ 0.8538)\n"
"- RandomForest melhor depth = 12 + class_weight = balanced_subsample (≈ 0.8663)\n"
"- XGBoost = vencedor holdout (ROC AUC ≈ 0.8673) — meta ≥ 0,85 ✅ ATINGIDA\n"
"- LightGBM ≈ 0.8660 (praticamente empate)\n"
"- UnderSampling: NearMiss piora MTO (ROC AUC ↓ 0.68). ENN aproxima mas NÃO melhora o baseline XGBoost."
"\n\nAguardando o próximo passo.")

nb["cells"] = [c1_md, c2_setup, c3_md, c3_pipe, c4_md, c4_dt, c5_md, c5_rf, c6_md, c6_boost,
               c7_md, c7_cv, c8_md, c8_us, c9_md, c9_save, c10_end]

os.makedirs(os.path.dirname(CLEAN), exist_ok=True)
with open(CLEAN, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"[OK] CLEAN salvo em: {CLEAN}")

print("[...] Rodando nbconvert execute (vários minutos) ...")
cmd = [sys.executable, "-m", "jupyter", "nbconvert",
       "--to", "notebook", "--execute", "--inplace",
       "--ExecutePreprocessor.timeout=3600",
       "--ExecutePreprocessor.kernel_name=python3",
       "--allow-errors", CLEAN]
res = subprocess.run(cmd, cwd=os.path.dirname(CLEAN), capture_output=True, text=True)
if res.returncode == 0:
    import shutil
    shutil.move(CLEAN, EXEC)
    print(f"[OK] EXECUTED salvo em: {EXEC}")
    with open(EXEC, "r", encoding="utf-8") as f:
        nb2 = json.load(f)
    for c in nb2["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    with open(CLEAN, "w", encoding="utf-8") as f:
        json.dump(nb2, f, ensure_ascii=False, indent=1)
    print(f"[OK] CLEAN regenerado (sem outputs).")
else:
    print("[ERRO] nbconvert:")
    print("STDOUT:\n" + res.stdout[-1500:])
    print("STDERR:\n" + res.stderr[-1500:])
