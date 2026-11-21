# Modelagem de risco de crédito | Bank

Abra `notebooks/credit_risk_modeling.ipynb` para consultar as análises, tabelas e figuras salvas no case. Para executar o estudo, o notebook importa o pacote local `credit_risk`. A regressão logística usa WoE monotônico e o LightGBM usa as variáveis selecionadas na escala original.

## Executar no Windows

Use Python 3.12 de 64 bits. O projeto já inclui o CSV do case. Extraia o ZIP, abra o PowerShell na pasta que contém `pyproject.toml` e execute os comandos abaixo. A chamada pelo executável do ambiente dispensa a ativação de scripts no PowerShell.

```powershell
winget install --id Python.Python.3.12 --exact --scope user
$python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $python312 --version
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m ipykernel install --user --name Bank-credit-risk --display-name "Python (Bank)"
.\.venv\Scripts\jupyter-lab.exe notebooks/credit_risk_modeling.ipynb
```

No JupyterLab, selecione o kernel **Python (Bank)**. O caminho padrão é `data/raw/DADOS_CASE_modelagem .csv`, com o espaço antes de `.csv`. Para ler outra cópia da base, você pode definir o caminho antes de iniciar o JupyterLab.

```powershell
$env:CREDIT_RISK_CSV = "C:\dados\DADOS_CASE_modelagem .csv"
```

Use **Restart Kernel and Run All Cells** para treinar e gerar os artefatos. A ordenação das linhas do CSV é parte do experimento. O notebook abre com as saídas históricas do arquivo recebido, inclusive a busca de 40 tentativas e a régua final. O ZIP inclui os modelos e a régua ajustados sobre o CSV fornecido, prontos para inferência no ambiente do projeto.

Em Linux/macOS, crie o ambiente com `python3.12 -m venv .venv`, ative-o com `source .venv/bin/activate` e use `python -m pip`, `python -m ipykernel` e `jupyter lab`. A trava em `requirements.lock` fixa as versões de execução e testes, incluindo condições de plataforma. O ambiente verificado foi Linux com Python 3.12.14. O notebook recebido registra Python 3.12.16 sem a lista completa de versões.

## Estrutura

| Caminho | Responsabilidade |
| --- | --- |
| `notebooks/credit_risk_modeling.ipynb` | Experimento em 16 seções, com parâmetros, chamadas, resultados e interpretação |
| `src/credit_risk/config` | Entrada do experimento e localização dos artefatos |
| `src/credit_risk/dataprep` | Ingestão, auditoria, correlação, seleção e amostragem estratificada |
| `src/credit_risk/binning` | Estratégias WoE, IV, monotonicidade e PSI |
| `src/credit_risk/modeling` e `optimization` | Logit não regularizado, LightGBM, espaço Hyperopt e histórico por tentativa |
| `src/credit_risk/evaluation` | Discriminação, calibração, bootstrap pareado, concentração, lift e captura |
| `src/credit_risk/explainability` e `visualization` | Contribuições SHAP, ranking global e figuras das análises |
| `src/credit_risk/rating` e `persistence` | Régua de sete grupos, pontuação e leitura/gravação de artefatos |
| `tests/` | Testes de componentes, fluxo sintético e regressão contra a referência |
| `reports/` | Inventário de células e funções, alterações, evidências de equivalência e referência numérica |

## Testes

No ambiente do projeto, os três comandos abaixo executam a suíte, o lint e a verificação de formatação. No Windows, use `.\.venv\Scripts\python.exe` no lugar de `python` quando o ambiente estiver desativado.

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

O teste `tests/regression/test_resultados_originais.py` descarta as saídas salvas de uma cópia do notebook, executa as células em um kernel novo e compara os resultados com `reports/referencia_original.json`. Ele usa o CSV incluído no projeto. A suíte também compara os 30 resultados intermediários e finais dos dois notebooks, tanto na base real como em um controle sintético, quando `CREDIT_RISK_ORIGINAL_NOTEBOOK` aponta para o arquivo `modelagem_risco-v1.ipynb` recebido. Consulte `reports/equivalencia.md` para a cobertura e as tolerâncias.

## Inferência com os modelos incluídos

Os arquivos são gravados em `artifacts/woe`, `artifacts/logistic`, `artifacts/lightgbm` e `artifacts/rating`. Use `CREDIT_RISK_ARTIFACTS` para direcionar uma execução a outra pasta. O exemplo abaixo recebe um DataFrame `novos_clientes` com as variáveis originais, sem exigir a target. As funções de carga localizam os artefatos incluídos no ZIP.

```python
from credit_risk.config.settings import ProjectPaths
from credit_risk.modeling.boosting import predict_pd
from credit_risk.persistence.models import load_lightgbm
from credit_risk.persistence.rating import load_rating_scale

paths = ProjectPaths.discover()
modelo = load_lightgbm(paths.lightgbm)
regua = load_rating_scale(paths.rating)
entradas = novos_clientes.loc[:, modelo.feature_name_]
resultado = regua.classify(predict_pd(modelo, entradas))
```

A coluna `Grupo_Risco` vale 1 para o maior risco. A classificação preserva a ordem das linhas e usa índice posicional. O código usa os limites de PD do discretizador ajustado, com sua precisão integral. Para a logística, recarregue `load_woe(paths.woe, "mono")` de `persistence.woe` e `load_logistic(paths.logistic)` de `persistence.models`, transforme os preditores em WoE e use `modeling.logistic.predict_logistic`.

Para examinar a explicação de um cliente da amostra SHAP ou a dependência de uma variável, o pacote fornece `plot_waterfall(shap_exp, row=0)` e `plot_dependence(shap_exp, feature="v6")` em `visualization.shap_plots`. As funções retornam a figura e os eixos. Use `fig.savefig(...)` para exportar uma figura.
