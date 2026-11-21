# Base de desenvolvimento

O arquivo `raw/DADOS_CASE_modelagem .csv` é uma cópia integral do CSV fornecido. O nome de origem contém um espaço antes de `.csv`. O leitor usa UTF-8 com BOM opcional, separador de campos `;` e vírgula decimal.

| Propriedade | Valor |
| --- | --- |
| Observações | 10.000 |
| Preditores | `v1` a `v8` |
| Desfecho | `target`, com 1.150 eventos e 8.850 não eventos |
| SHA-256 | `594182b6892054de80c8cc04283aebfd8568bb20b596ead073973260a6eca2a2` |

Preserve a ordem das linhas para reproduzir o split estratificado. A variável de ambiente `CREDIT_RISK_CSV` permite apontar o notebook para outra cópia do arquivo. O teste de integração constrói a própria amostra sintética em diretório temporário.
