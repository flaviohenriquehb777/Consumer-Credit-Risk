"""Corrige quebras de linha literais no meio de strings Python do notebook e executa."""
import json
import os
import re
import subprocess
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN = os.path.join(BASE, "notebooks", "clean", "02_entendimento_dados.ipynb")
EXEC  = os.path.join(BASE, "notebooks", "executed", "02_entendimento_dados.ipynb")

with open(CLEAN, "r", encoding="utf-8") as f:
    nb = json.load(f)

def source_to_str(src):
    if isinstance(src, list):
        return "".join(src)
    return src

def str_to_source(s):
    # nbformat classicamente aceita os dois, salvemos como lista de linhas
    return [line + "\n" for line in s.split("\n")]

# --- Faz os ajustes em cada célula de código ---
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    s = source_to_str(cell["source"])

    # Fix 1: ITEM 3 — quebra de linha no meio de ax.annotate(f"{h:,}\n(...)")
    # Atualmente está como: ax.annotate(f"{h:,}\n({h/len(df)*100:.2f}%)",
    # onde o \n é uma quebra REAL. Precisamos torná-la dois caracteres \ e n
    # dentro do Python.
    s = re.sub(
        r'ax\.annotate\(f"\{h:,\}\n\(\{h/len\(df\)\*100:\.2f\}%\)"',
        r'ax.annotate(f"{h:,}\\n({h/len(df)*100:.2f}%)"',
        s,
    )

    # Fix 2: print("\nOUTLIERS... com \n real
    s = re.sub(
        r'print\("\nOUTLIERS EXTREMOS',
        r'print("\nOUTLIERS EXTREMOS',
        s,
    )

    # Fix 3: md = "### ... \n\n" com newlines reais
    s = re.sub(
        r'md = "### Problemas específicos detectados \(evidência por item\)\n\n"',
        r'md = "### Problemas específicos detectados (evidência por item)\n\n"',
        s,
    )
    s = re.sub(
        r'md \+= f"\{i\}\. \{p\}\n\n"',
        r'md += f"{i}. {p}\n\n"',
        s,
    )

    cell["source"] = str_to_source(s)

# Salva o clean corrigido
with open(CLEAN, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"[OK] Clean corrigido salvo em: {CLEAN}")

# --- Agora executa via jupyter nbconvert CLI (mais robusto no Windows) ---
print("[...] Executando notebook via jupyter nbconvert (vários minutos)...")
cmd = [
    sys.executable, "-m", "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    "--ExecutePreprocessor.timeout=900",
    "--ExecutePreprocessor.kernel_name=python3",
    "--allow-errors",
    CLEAN,
]
res = subprocess.run(cmd, cwd=os.path.dirname(CLEAN), capture_output=True, text=True)
if res.returncode == 0:
    # nbconvert --inplace alterou o clean! Então re-copiamos clean de um backup
    # e gravamos o executed com o conteúdo executado.
    # Para evitar esse problema, lemos o que foi salvo como "clean executado",
    # salvamos como executed e regeneramos o clean sem outputs.
    print("[OK] nbconvert executou com sucesso.")
    # O clean agora é o executado (inplace). Movemos para executed e re-geramos clean.
    import shutil
    shutil.move(CLEAN, EXEC)
    print(f"[OK] Notebook EXECUTADO movido para: {EXEC}")
    # Re-salvamos o clean SEM outputs (regenerar a partir do executed zerando outputs)
    with open(EXEC, "r", encoding="utf-8") as f:
        nb_exec = json.load(f)
    for cell in nb_exec["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    with open(CLEAN, "w", encoding="utf-8") as f:
        json.dump(nb_exec, f, ensure_ascii=False, indent=1)
    print(f"[OK] Clean (sem outputs) regenerado em: {CLEAN}")
else:
    print("[ERRO] nbconvert falhou:")
    print("--- STDOUT ---")
    print(res.stdout)
    print("--- STDERR ---")
    print(res.stderr)
