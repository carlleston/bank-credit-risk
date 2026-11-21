# Inventário da execução histórica

Fonte: `modelagem_risco-v1.ipynb`. O índice da primeira célula é 0. A coluna `In` registra o contador de execução gravado no arquivo. A célula 57 registra `In[57]` depois de `In[42]` na célula 55. Esse salto indica execuções intermediárias na sessão.

São 61 células, incluindo 15 de Markdown e 46 de código. Três células de código no final estão vazias. As saídas contêm 37 tabelas HTML e 23 figuras PNG. O kernel registrado usava Python 3.12.16. As versões das bibliotecas e um hash do CSV não constam do arquivo.

## Células e dependências

Os números nas entradas referem-se às células de origem. As células Markdown foram identificadas pelo título recebido. Os módulos indicados estão sob `src/credit_risk/`.

| Célula | In | Entradas | Transformação ou saída | Destino |
|---:|---:|---|---|---|
| 0 | — | — | Lendo os dados | Markdown |
| 1 | 1 | CSV | df, dimensões e tipos | `dataprep.ingestion` |
| 2 | 2 | df (1) | estatísticas descritivas | `DataFrame.describe` |
| 3 | — | — | Correlação | Markdown |
| 4 | — | — | Correlação entre as variaveis | Markdown |
| 5 | 3 | df (1) | corr_pearson, correlacao_target, mapa Pearson | `dataprep.selection`, `visualization.eda` |
| 6 | — | — | Correlação entre variaveis e target | Markdown |
| 7 | 4 | df (1) | corr_target, mapa Pearson/Spearman por variável | `dataprep.selection`, `visualization.eda` |
| 8 | — | — | Removendo variaveis que possui uma multicolinearidade alta | Markdown |
| 9 | 5 | df (1) | selecionadas, removidas, df_filtrado | `dataprep.selection` |
| 10 | — | — | Separando em treino e Test | Markdown |
| 11 | 6 | df_filtrado (9) | X_train, X_test, y_train, y_test e proporções | `dataprep.splitting` |
| 12 | — | — | Criando transformação Woe | Markdown |
| 13 | 7 | split (11) | binning_mono, binning_livre, quatro X transformados, comparacao_iv, persistência | `binning.woe`, `persistence.woe` |
| 14 | 8 | variaveis e binning_mono/livre (13) | 14 tabelas de bins, duas por preditor | `binning.woe`, `visualization.eda` |
| 15 | — | — | Estabilidade | Markdown |
| 16 | 9 | séries WoE de treino/teste | função calcular_psi_woe | `binning.stability` |
| 17 | 10 | quatro X transformados (13), função (16) | df_psi | `binning.stability` |
| 18 | 11 | df_psi (17) | função classificar_psi, df_psi com status | `binning.stability` |
| 19 | 12 | df_psi (18) | gráfico de estabilidade | `visualization.eda` |
| 20 | — | — | Modelagem | Markdown |
| 21 | — | — | Modelo linear | Markdown |
| 22 | 13 | WoE monotônico (13), y_train/y_test (11) | resultado_lr, tabela_coeficientes, y_proba_train/test, resultados, ROC | `modeling.logistic`, `evaluation.discrimination`, `visualization.modeling`, `persistence.models` |
| 23 | — | — | Avaliação da calibração | Markdown |
| 24 | 14 | y_train/test (11), PD logística (22) | resultados_calibracao, curva de calibração | `evaluation.calibration`, `visualization.comparison` |
| 25 | 15 | y_test (11), y_proba_test (22) | tabela_decil | `evaluation.calibration` |
| 26 | — | — | Modelo Não-Linear | Markdown |
| 27 | 16 | WoE livre (13), y_train/test (11) | X_train_lgb/X_test_lgb de prévia, distribuição de y | `modeling.boosting` |
| 28 | 17 | X_train/X_test, y_train/y_test (11) | X_train_lgb/X_test_lgb efetivos e y_train_lgb/y_test_lgb | `modeling.boosting` |
| 29 | 18 | matriz, target e seed (28), converter_parametros (31) ao invocar | objective, médias por fold e mediana de best_iteration | `optimization.hyperopt_search`, `modeling.boosting` |
| 30 | 19 | distribuições Hyperopt | space | `optimization.hyperopt_search` |
| 31 | 20 | parâmetros amostrados | converter_parametros | `optimization.hyperopt_search` |
| 32 | 21 | objective/space/conversão (29–31), dados (28) | trials, best_params, melhor_trial, best_n_estimators | `optimization.hyperopt_search` |
| 33 | 22 | trials (32) | historico, gráfico das AUCs | `optimization.hyperopt_search`, `visualization.modeling` |
| 34 | 23 | matriz e target (28), parâmetros e árvores (32) | modelo_lgb, y_proba_train_lgb, y_proba_test_lgb | `modeling.boosting` |
| 35 | 24 | targets (28), PD (34) | resultados_lgb | `evaluation.discrimination` |
| 36 | 25 | resultados_lgb (35) | gráfico de performance | `visualization.modeling` |
| 37 | 26 | targets (28), PD (34), métricas (35) | ROC e duas curvas de KS | `evaluation.discrimination`, `visualization.modeling` |
| 38 | 27 | modelo_lgb (34), X_test_lgb (28) | amostra de teste, contribuições e shap_exp | `explainability.contributions` |
| 39 | 28 | shap_exp (38) | beeswarm | `visualization.shap_plots` |
| 40 | 29 | modelo (34), parâmetros (32), métricas (35), histórico (33), contribuições (38) | modelos/objetos serializados e importância média absoluta | `explainability.contributions`, `persistence.models` |
| 41 | — | — | Calibração | Markdown |
| 42 | 30 | modelo e PD (34), matrizes e targets (28) | resultados_calibracao_lgb | `evaluation.calibration` |
| 43 | 31 | PD e targets (42, 28) | curvas de calibração | `evaluation.calibration`, `visualization.comparison` |
| 44 | 32 | y_proba_test_lgb, y_test_lgb (42, 28) | tabela_decil_lgb | `evaluation.calibration` |
| 45 | 33 | tabela_decil_lgb (44) | gráfico por decil | `visualization.comparison` |
| 46 | 34 | y_proba_test_lgb, y_test_lgb (42, 28) | regressão de calibração, tabela_calibracao | `evaluation.calibration` |
| 47 | 35 | resultado_lr/X_test_sm (22), modelo_lgb/X_test_lgb (34, 28), y_test (11) | proba_lr, proba_lgb, y_true, comparacao_modelos | `modeling.logistic`, `modeling.boosting`, `evaluation.comparison` |
| 48 | 36 | comparacao_modelos (47) | comparação AUC/Gini/KS | `visualization.comparison` |
| 49 | 37 | comparacao_modelos (47) | comparação Brier/Log Loss | `visualization.comparison` |
| 50 | 38 | proba_lr, proba_lgb, y_true (47) | ROC comparativa no teste | `evaluation.comparison`, `visualization.comparison` |
| 51 | 39 | proba_lr, proba_lgb, y_true, comparacao_modelos (47) | diferenças pareadas, IC, diferença observada, histograma | `evaluation.bootstrap`, `visualization.comparison` |
| 52 | — | — | Concetração dos modelos | Markdown |
| 53 | 40 | proba_lr, proba_lgb, y_true (47) | HHI por eventos, tabela_lr, tabela_lgb, comparacao_hhi | `evaluation.concentration` |
| 54 | 41 | tabela_lr, tabela_lgb (53) | comparacao_eventos e gráfico de participação | `evaluation.concentration`, `visualization.comparison` |
| 55 | 42 | PD (47), HHI de eventos (53), faixas_pd | HHI de carteira e comparacao_hhi_final | `evaluation.concentration` |
| 56 | — | — | Lift | Markdown |
| 57 | 57 | modelo (34), matrizes e targets (28), mapa acordado | binning_pd, regua_pd, clientes, tabelas de treino/teste, comparação, lift e figuras | `rating.scale`, `evaluation.lift`, `visualization.rating`, `persistence.rating` |
| 58 | — | — | Célula vazia | Sem operação |
| 59 | — | — | Célula vazia | Sem operação |
| 60 | — | — | Célula vazia | Sem operação |

## Funções e redefinições

| Definição no arquivo recebido | Células | Implementação no pacote |
|---|---|---|
| `calcular_psi_woe` | 16 | `binning.stability.calcular_psi_woe` |
| `classificar_psi` | 18 | `binning.stability.classificar_psi` |
| `avaliar_modelo` | 22, 35, 47 | `evaluation.discrimination.discrimination_metrics`, `evaluation.comparison.compare_predictions` |
| `avaliar_calibracao` | 24, 42 | `evaluation.calibration.calibration_summary` |
| `objective` | 29 | `optimization.hyperopt_search.evaluate_trial` |
| `converter_parametros` | 31 | `optimization.hyperopt_search.typed_params` |
| `plot_ks` | 37 | `visualization.modeling.plot_ks` |
| `calcular_hhi` | 53 | `evaluation.concentration.hhi` |
| `hhi_eventos` | 53 | `evaluation.concentration.event_hhi` |
| `hhi_carteira` | 55 | `evaluation.concentration.portfolio_hhi` |
| `aplicar_rating` | 57 | `rating.scale.RatingScale.classify` |
| `calcular_tabela_risco` | 57 | `evaluation.lift.risk_table` |
| `verificar_monotonicidade` | 57 | `evaluation.lift.is_monotonic` |
| `plotar_taxas` | 57 | `visualization.rating.plot_rates` |
| `plotar_lift` | 57 | `visualization.rating.plot_lift` |

A célula 29 usa `RANDOM_STATE`, `X_train_lgb`, `y_train_lgb` e a função da célula 31 por resolução de nomes no momento da chamada. A célula 55 captura `faixas_pd`. A célula 57 captura `n_bins`, `N_GRUPOS` e `MAPA_RATING`. A montagem do modelo depende da ordem das colunas. O HHI, os decis, o bootstrap e as tabelas de rating dependem da ordem das linhas e das previsões.

## Parâmetros originais

| Etapa | Configuração executada |
|---|---|
| Entrada e target | CSV em UTF-8 com BOM aceito, separador `;`, decimal `,`, target binária `target` e evento 1. |
| Seleção | Em toda a base, ordem decrescente de Pearson absoluto com o target. Descarta o candidato se qualquer preditor já mantido apresentar correlação absoluta ≥ 0,80. |
| Partição | 80% treino e 20% teste, estratificados pelo target, `random_state=42`. |
| WoE | Dois BinningProcess com máximo de 5 bins, mínimo de 2% por bin e 1% por prebin, solver CP. Zero especial em todos os preditores. Tendências `auto_asc_desc` e `None`. |
| PSI | Frequência por valor WoE, união das categorias, adição de 1e-6 e renormalização. Baixa abaixo de 0,10, moderada de 0,10 a menos de 0,25, elevada a partir de 0,25. |
| Logística | `statsmodels.Logit`, constante adicionada explicitamente, `maxiter=2000`, IC 95%, coeficientes e razões de chances em WoE monotônico. |
| CV | 3 folds estratificados com embaralhamento e seed 42. LightGBM binário com máximo de 1.000 árvores, `binary_logloss`, parada após 50 rodadas, primeira métrica, `subsample_freq=1`, `n_jobs=-1`. |
| Hyperopt | TPE, 40 tentativas, RNG NumPy 42. Objetivo é menor log loss médio entre folds. AUC é uma medida secundária do trial vencedor. Número de árvores é a mediana das melhores iterações desse trial, convertida em inteiro. |
| Espaço de busca | Discretas uniformes: folhas 4–31 (passo 1), profundidade 2–8 (1), mínimo de observações 20–150 (5). Uniformes: amostragem de linhas e de colunas 0,65–1. Log-uniformes: aprendizado 0,01–0,15 e penalizações L1/L2 0,001–10. |
| Métricas | Gini = 2 × AUC − 1, KS = máximo(TPR − FPR). Decis em risco crescente por `qcut(rank(method="first"))`. Curvas de calibração com dez bins quantílicos. Regressão de calibração com intercepto, logit da PD truncada em [1e-8, 1−1e-8] e IC Wald de 95%. |
| Comparação | Bootstrap com reposição, 2.000 réplicas, seed 42, amostras estratificadas por evento, mesmos índices nos dois modelos. IC percentil 2,5–97,5%. |
| Concentração | HHI como soma dos quadrados das participações. Eventos por decil, clientes em faixas de PD com limites 0, 0,01, 0,03, 0,05, 0,10, 0,15, 0,20, 0,30, 0,50 e 1. |
| SHAP | Amostra de até 1.000 observações do teste, seed 42. `pred_contrib=True` no booster, valor base na última coluna e até 15 variáveis no beeswarm. |
| Rating | PD do LightGBM no treino, OptimalBinning ascendente, solver CP, até 10 bins, mínimo de 5% por bin e de 1% por prebin. A régua acordada exige exatamente 8 bins originais. Invertem-se os índices e unem-se os grupos 3 e 4, nessa ordem, formando 7 classes, com o maior risco no grupo 1. Monotonicidade admite tolerância absoluta de 1e-12. |

## Interpretação e pontos de validação

| Comportamento do notebook recebido | Consequência | Avaliação separada proposta |
|---|---|---|
| A seleção usa o target da amostra completa antes do split. | O conjunto de preditores incorpora informação dos rótulos do teste. A comparação histórica preserva essa ordem. | Pré-separar treino e teste e repetir a seleção exclusivamente no treino. Na avaliação por CV, refazer seleção supervisionada em cada fold. Comparar em uma base de validação adicional. |
| A célula 27 usa WoE livre apesar do comentário sobre variáveis originais. A 28 substitui essa matriz pelos preditores em sua escala original antes da busca. | O LightGBM e as cores do gráfico SHAP usam preditores originais. O Logit usa WoE monotônico. | Em um experimento com boosting sobre WoE, ajustar o binning separadamente no treino de cada fold. |
| Zero recebe o tratamento de código especial mesmo em variáveis nas quais pode representar um valor legítimo. O código usa os defaults da biblioteca na transformação de categorias especiais/ausentes. | A tabela dos bins apresenta o WoE empírico do grupo especial. A política usada para transformar os valores deve ser preservada ao pontuar novos clientes. | Confirmar o significado de zero com o dicionário de dados antes de alterar o binning. |
| A avaliação de teste usa uma única partição aleatória. O PSI compara essa partição com o treino. | PSI baixo caracteriza essa divisão, sem medir deriva entre safras. A melhora de Gini no bootstrap é condicional aos modelos já treinados e ao teste disponível. | Avaliar vintages futuras e, havendo múltiplas observações por tomador, separar por cliente. |
| Os bins de rating usam PD do modelo ajustado e eventos do mesmo treino. O mapa de 8 para 7 já chega definido no código. | A monotonicidade observada no teste é empírica. O arquivo não identifica a base que orientou a escolha histórica de consolidar 3 e 4. | Aplicar a régua congelada a uma terceira base ou safra e investigar estabilidade dos grupos adjacentes. |
| Os decis das tabelas separam empates por ordem de aparição. A curva de calibração usa quantis da PD diretamente. | Em amostras com empates, a curva pode ter pontos e massas diferentes dos dez decis tabulados. | Registrar ambos os métodos ao comparar gráficos e tabelas. |
| Intercepto e inclinação da calibração são estimados no teste para diagnóstico. O modelo de calibração não transforma a PD utilizada na régua. | O/E próximo de 1 pode coexistir com desvios por decil. | Avaliar recalibração em base própria, reservando outra amostra para aferição. |
| O código chama o resumo completo do Logit em uma expressão intermediária. O formato de saída gravado contém os coeficientes, p-valores, odds ratios e intervalos. | As tabelas visíveis servem de referência numérica para a regressão. | Consultar o objeto ajustado para recuperar o resumo integral. |

## Como ler a referência numérica

`referencia_original.json` guarda o SHA-256 do notebook recebido, as células e posições de saída, as tabelas com cabeçalhos e índices, a contagem de figuras e os escalares extraídos. As contagens de eventos são a soma dos eventos nos sete grupos de rating gravados. Os parâmetros vencedores e o log loss de CV vêm do texto do Hyperopt, os três valores do bootstrap vêm do texto da célula 51.

A regressão verifica igualdade de rótulos, ordenação, contagens de clientes/eventos e grupos originais. Para tabelas numéricas, usa tolerância absoluta de meia unidade na última casa divulgada (quatro, cinco ou seis casas), com margem de ponto flutuante de duas partes por milhão dessa tolerância e tolerância relativa zero. Para parâmetros do Hyperopt e log loss exibidos em precisão estendida, a tolerância absoluta é 1e-12. Intervalos rotulados são comparados como texto. O arquivo original não publicou predições linha a linha em precisão integral.

O teste limpa as saídas embutidas antes da execução e exige que as 23 figuras históricas voltem a ser emitidas nas etapas correspondentes. O conteúdo numérico de tabelas/indicadores é verificado separadamente. A renderização gráfica pode variar com backend, fontes e versão do Matplotlib.
