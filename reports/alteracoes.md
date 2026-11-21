# Alterações no projeto

| Componente | Implementação no projeto |
| --- | --- |
| Notebook | Apresentação em 16 seções. As células de código configuram o experimento, chamam o pacote local e apresentam as saídas. O campo de metadados `original_cell` associa cada resultado histórico ao bloco que o produziu. |
| Dados | Leitura com os mesmos parâmetros e validação de schema. A nova tabela de auditoria descreve ausentes, zeros e observações além das cercas de Tukey; a contagem de linhas repetidas é informativa. |
| WoE | Uma fábrica configura o mesmo `BinningProcess` para duas estratégias de tendência. Cada objeto ajustado é aplicado ao treino e ao teste. Valores especiais e ordem de colunas seguem o código de origem. |
| Ajuste de modelos | A logística conserva o Logit do statsmodels com intercepto e sem regularização. A matriz exploratória de WoE livre e a matriz original usada no LightGBM receberam nomes distintos. |
| Busca | A configuração do experimento é passada explicitamente. A função de construção do LightGBM atende treino final e validação. A busca substitui o uso implícito de variáveis globais e permite fornecer uma fábrica de transformadores para ajuste por fold em experimentos separados. |
| Métricas e figuras | Funções compartilhadas calculam AUC, Gini, KS, calibração e concentração. Os gráficos recebem dados ou resultados calculados e devolvem figura e eixos para exibição ou exportação. |
| Persistência | As subpastas de `artifacts/` agrupam WoE, logística, LightGBM e rating. Os nomes históricos dos arquivos são mantidos. O modelo logístico e a tabela detalhada de tentativas são artefatos adicionais. |
| Inferência | O discretizador ajustado é recarregado juntamente com o mapa de consolidação e os limites. Uma nova PD pode ser classificada sem informar a target. |
| Verificação | A suíte distingue testes de componentes, integração com CSV sintético, comparação pareada entre notebooks e regressão numérica contra a referência histórica. |

## Decisões de apresentação

A explicação do lift antecede a construção da régua. As respectivas tabelas e figuras aparecem depois do ajuste dos cortes. A análise de PSI passou à seção de estabilidade, e a calibração dos dois modelos foi reunida em uma seção. Essas mudanças respeitam as dependências de cálculo.

Mensagens operacionais da execução original continuam visíveis entre as saídas históricas. Ao reexecutar, a apresentação prioriza tabelas e gráficos. As rotinas de persistência podem registrar o término de uma gravação pelo logger do módulo. O artefato `binning_pd_monotonico.pkl` usa `pickle`, como na versão recebida da última célula.

O inventário detalha os parâmetros e ressalvas metodológicas. A seleção supervisionada anterior à divisão e a regra 3+4 foram reproduzidas. Eventuais estudos com seleção dentro dos folds ou em uma safra posterior constituem novos experimentos.
