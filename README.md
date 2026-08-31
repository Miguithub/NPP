# NPP

Implementación exploratoria del principio de proporcionalidad anidada (NPP) aplicada a volúmenes transaccionales de criptomonedas.

## Primer acercamiento: modelo de un solo agente

> **Nota metodológica:** este primer acercamiento modela una sola variable del simplex: la cuota futura de Bitcoin como agente individual.

El notebook principal estudia la cuota probabilística futura de Bitcoin dentro del volumen económico sistémico. Incluye:

- construcción del simplex probabilístico;
- factores de resistencia;
- información mutua y selección temporal de variables;
- benchmarks de persistencia y modelos autoregresivos;
- ARX y XGBoost;
- ablación por familias;
- evaluación walk-forward.

Archivo: `Primer_acercamiento_modelo_un_solo_agente.ipynb`.

## Modelo GAT multivariado

La rama `codex/modelo-gat` incorpora el primer modelo conjunto del simplex completo. Cada criptoactivo es un nodo y el GAT predice simultáneamente la redistribución de las cuotas a un horizonte de diez minutos, con cortes cronológicos, purga, escalado exclusivo de train, grafo CLR causal y `shuffle=False`.

Archivos principales:

- `Modelo_GAT_NPP.ipynb`: notebook ejecutado y auditable;
- `modelo_gat_npp.py`: fuente sincronizada y reproducible;
- `NOTA_CIENTIFICA_CABEZA_ENERGETICA_VS_GAT_NPP.md`: distinción formal y empírica entre la cabeza predictiva regularizada y la solución energética estricta.

Este repositorio contiene investigación exploratoria y no constituye asesoramiento financiero.

## Contraste inicial de tres semillas

La rama `codex/contraste-tres-semillas` incorpora el primer análisis de estabilidad del modelo GAT con semillas `11`, `23` y `42` sobre el mismo test cronológico.

Resultados principales:

- GAT + NPP obtuvo la mejor media en CE, KL, MAE y RMSE;
- superó al GAT libre en MAE y RMSE en las tres semillas;
- la cabeza energética perdió levemente en métricas predictivas, pero mejoró Aitchison frente a la salida principal NPP en las tres semillas;
- todos los GAT superaron ampliamente a persistencia en CE, KL, JS, MAE y RMSE;
- tres semillas validan el pipeline, pero no bastan para afirmar robustez estadística.

Limitación metodológica central: el denominado GAT libre comparte el grafo híbrido con 25% de proximidad energética. Por ello, el contraste actual mide el aporte marginal de la loss NPP sobre una topología ya informada por energía. La siguiente ablación debe incorporar un GAT puro con aristas exclusivamente CLR.

Archivos:

- `GAT_crypto_contraste_experimental.ipynb`;
- `NOTA_CIENTIFICA_CONTRASTE_TRES_SEMILLAS.md`;
- `resultados/contraste_3_semillas/metricas_globales_por_semilla.csv`;
- `resultados/contraste_3_semillas/contraste_emparejado_resumen.csv`;
- `resultados/contraste_3_semillas/resumen_por_modelo.csv`.
