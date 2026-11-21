# Equivalência dos resultados na base do case

O notebook original e a versão modular foram executados do início ao fim, em kernels independentes, com a mesma cópia do CSV fornecido. Em cada uma das 10.000 linhas, as probabilidades produzidas por cada modelo coincidiram exatamente entre as duas implementações. Os sete grupos de rating também foram atribuídos de forma idêntica.

## Verificações realizadas

| Verificação | Resultado |
| --- | --- |
| Entrada | 10.000 registros em sua ordem original, 1.150 eventos. O arquivo em `data/raw/` é idêntico ao CSV recebido. O SHA-256 está em `data/README.md`. |
| Seleção e split | Sete preditores na mesma ordem. Treino com 8.000 registros e 920 eventos. Teste com 2.000 registros e 230 eventos. Índices e targets coincidiram exatamente. |
| Estimação | Quatro matrizes WoE, IV, PSI, coeficientes do Logit, parâmetros do LightGBM e probabilidades de treino e teste coincidiram em precisão integral. |
| Busca | 40 tentativas com três folds estratificados. A tentativa de menor log loss média (0,2712537925858644) resultou em 67 árvores e AUC média de 0,81338249. |
| Avaliação e explicabilidade | O comparador verificou 30 objetos científicos, incluindo tabelas de calibração e discriminação, HHI, valores SHAP e bases das explicações. Todos coincidiram em precisão integral. |
| Rating | Os cortes aprendidos, a consolidação dos grupos originais 3 e 4, as tabelas e as classificações individuais foram idênticos entre as implementações. |
| Saídas do notebook | As duas novas execuções geraram as 23 figuras e as 37 tabelas históricas nas respectivas etapas. As tabelas recalculadas coincidiram entre si e passaram na regressão contra o histórico, com as tolerâncias de arredondamento descritas no inventário. |
| Inferência após carga | As duas transformações WoE, os dois modelos e a régua foram carregados dos arquivos salvos. Reaplicar os objetos a treino e teste reproduziu as matrizes, PDs e classificações da execução em memória. |
| Controle sintético | A comparação independente sobre 6.400 registros também obteve igualdade exata dos 30 objetos científicos. |
| Ambiente | Python 3.12.14 com as versões de `requirements.lock`. O notebook histórico indica Python 3.12.16. O verificador de dependências do pip, o lint e a formatação da base Python passaram. |

O notebook do ZIP mantém os 96 objetos de saída históricos exatamente como registrados e inclui os resultados da nova auditoria de qualidade. Na comparação das imagens geradas pelas duas novas execuções, 22 PNGs foram idênticos. O *beeswarm* do SHAP distribui aleatoriamente pontos que se sobrepõem, o que alterou sua disposição visual. As contribuições numéricas de cada observação explicada e os valores de base coincidiram exatamente.

A auditoria identificou uma repetição de linha completa além da primeira ocorrência. A seleção de variáveis e o treinamento usaram as 10.000 linhas, seguindo o procedimento original.

A suíte completa passou nos **20 testes**, incluindo a regressão histórica e as duas comparações pareadas entre notebooks.

## Desempenho reproduzido no teste

| Métrica | Regressão logística | LightGBM |
| --- | ---: | ---: |
| AUC | 0,8255 | 0,8434 |
| Gini | 0,6510 | 0,6868 |
| KS | 0,5019 | 0,5476 |
| Brier | 0,0783 | 0,0758 |
| Log loss | 0,2700 | 0,2603 |
| Observado/esperado | 0,9974 | 1,0127 |

O ganho de Gini do LightGBM foi de 0,0357. O intervalo de confiança percentil, calculado em 2.000 réplicas de bootstrap pareado, foi de 0,0128 a 0,0589. O grupo de maior risco reuniu 101 clientes e 70 dos 230 eventos de teste, com lift de 6,03.

## Reproduzir a validação

Com o ambiente instalado conforme o README, execute `python -m pytest tests/regression/test_resultados_originais.py -q`. Esse teste limpa as saídas gravadas de uma cópia temporária do notebook, usa o CSV distribuído, treina os modelos e compara os resultados com `reports/referencia_original.json`. As tolerâncias de apresentação são proporcionais às casas decimais publicadas. As contagens e os rótulos são conferidos exatamente.

Para repetir a comparação em precisão integral entre as implementações, aponte `CREDIT_RISK_ORIGINAL_NOTEBOOK` para o notebook original recebido. No PowerShell, por exemplo, use o comando abaixo com o caminho real do seu arquivo.

```powershell
$env:CREDIT_RISK_ORIGINAL_NOTEBOOK = "C:\dados\modelagem_risco-v1.ipynb"
.\.venv\Scripts\python.exe -m pytest -q
```

Essa configuração habilita os dois testes pareados opcionais, um com a base real e outro com a sintética. O notebook externo é usado exclusivamente como implementação de referência nesses testes. A execução normal do projeto utiliza o notebook modular e os arquivos incluídos no ZIP.
