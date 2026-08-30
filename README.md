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
