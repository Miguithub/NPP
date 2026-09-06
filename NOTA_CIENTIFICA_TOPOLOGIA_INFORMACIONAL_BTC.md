# Nota científica: utilidad de `BTC_y` como topología informacional de régimen

## 1. Alcance

Este experimento traduce a un pipeline reproducible el recorrido documentado en el chat compartido [«Acceso a Drive»](https://chatgpt.com/share/6a9da6eb-5c68-83e9-8b4d-c3c4109c5ff1). Su pregunta no es si `BTC_y` predice mecánicamente el precio de Bitcoin, sino **qué tipo de información económica contiene la posición relativa de BTC dentro del simplex transaccional**.

`BTC_y` es una coordenada composicional: representa participación o probabilidad relativa y sólo adquiere significado respecto de la masa total del sistema. En este sentido operativo se la denomina *topología informacional*. El término no implica que este script pruebe por sí solo una topología matemática completa ni el NPP causal: estudia una coordenada del estado sistémico y su utilidad predictiva incremental.

## 2. Hipótesis que organiza el análisis

La secuencia inicial:

\[
BTC_y(t) \rightarrow BTCUSDT(t+h)
\]

se reemplaza, después de las pruebas de descarte, por:

\[
BTC_y(t) \rightarrow P\!\left(Z^{vol}_{t+h}=k\right).
\]

La hipótesis central es que `BTC_y` funciona mejor como **descriptor de estado o variable de gating** que como regresor directo de precio, retorno o volatilidad puntual.

## 3. Evidencia obtenida

El análisis previo encontró:

- correlación contemporánea casi nula entre `BTC_y` y `BTCUSDT` (Pearson `-0.01754`, Spearman `-0.05768`);
- dependencia no lineal pequeña pero positiva frente al precio futuro, detectada por información mutua, que en gran medida desaparecía al controlar por el nivel de precio;
- ausencia de señal útil de `ΔBTC_y` para dirección futura (`AUC ≈ 0.5012`) y para retorno esperado;
- heterocedasticidad asociada a regiones de `ΔBTC_y(t-4)`, con `p` de Levene cercano a `0.00188`;
- caída de la información mutua al normalizar el retorno por volatilidad pasada, lo que vinculó el hallazgo con el **régimen de volatilidad** y no con la dirección;
- ganancia prácticamente nula al sumar `ΔBTC_y` a una regresión directa de volatilidad;
- señal propia para clasificar tres regímenes: sólo `BTC_y` y su diferencia lograron balanced accuracy de `41.8%`, frente a `33.3%` balanceado;
- en el árbol combinado de tres regímenes, Macro-F1 pasó de `0.4406` a `0.4650`; en walk-forward pasó de `0.4083` a `0.4233` y mejoró en cinco de cinco folds;
- la ingeniería multiescala de `BTC_y` aportó poco en tres regímenes, pero fue material al separar extremos: con cinco regímenes y colas del `4.5%`, Macro-F1 holdout pasó de `0.2588` a `0.2895` (`+11.9%` relativo), y walk-forward de `0.2304` a `0.2562` (`+11.2%`), ganando en cinco de cinco folds;
- en cinco quintiles libres la mejora fue intermedia: `+7.4%` relativo en holdout y `+4.8%` en walk-forward.

Estos valores son antecedentes del chat, no resultados precargados por el nuevo script. El pipeline vuelve a estimarlos sobre los archivos proporcionados y exporta todas las métricas con su modelo y split explícitos.

## 4. Traducción experimental A–K

El archivo `analisis_topologia_informacional_btc.py` implementa:

| Bloque | Pregunta | Salida principal |
|---|---|---|
| A | ¿Existe asociación contemporánea entre `BTC_y` y precio? | Pearson, Spearman e información mutua |
| B–C | ¿La aparente relación con el nivel futuro sobrevive al control por precio actual? | dependencia y correlación parcial |
| D | ¿`ΔBTC_y` se relaciona con nivel, cambio o retorno? | métricas por target y lag |
| E | ¿Hay lags informativos? | barrido causal de lags |
| F | ¿El candidato de lag informa dirección, magnitud, colas o varianza? | AUC, MI, Levene y retorno estandarizado |
| G | ¿La señal apunta a volatilidad futura y a qué horizonte? | matriz lag × horizonte |
| H | ¿Agrega precisión en regresión una vez conocida la volatilidad pasada? | holdout y walk-forward de regresión |
| I | ¿Sirve para tres regímenes? | dummy, árboles y boosting anidado |
| J | ¿Su utilidad crece con cinco regímenes y colas explícitas? | holdout, métricas por clase y walk-forward |
| K | ¿Cuánto agrega la topología frente a un control equivalente? | contrastes emparejados por arquitectura |

## 5. Identificación de la utilidad informacional

Comparar un árbol simple con un boosting de muchas variables no identifica el aporte de `BTC_y`: mezcla información nueva, capacidad del modelo y dimensionalidad. Por eso el script incorpora pares anidados:

1. `tree_topology` frente a `dummy_most_frequent`: señal propia mínima;
2. `tree_volatility_plus_topology` frente a `tree_volatility`: aporte incremental en árbol;
3. `advanced_volatility_plus_topology` frente a `advanced_volatility_multiscale`: aporte incremental bajo el mismo boosting;
4. `advanced_topology_multiscale`: capacidad autónoma de la familia topológica.

La utilidad se define con la orientación correcta:

\[
Gain_{F1}=F1_{con\ topología}-F1_{control},
\]

\[
Gain_{LogLoss}=LogLoss_{control}-LogLoss_{con\ topología}.
\]

En ambos casos un valor positivo favorece la incorporación de la topología. El resultado se guarda en `13_utilidad_topologia.csv` y se resume por régimen en `14_resumen_utilidad_topologia.csv`.

## 6. Contrato temporal y ausencia de leakage

- No hay `shuffle`.
- Las features rolling, EMA, momentum y lags sólo utilizan información hasta `t`.
- La volatilidad futura usa `t+1, …, t+h` únicamente como target.
- El 30% final se conserva como holdout cronológico.
- Entre train y test se purgan `h` observaciones para evitar que targets del train invadan el test.
- Cada fold walk-forward utiliza una ventana de entrenamiento expansiva y `gap=h`.
- Los umbrales de los regímenes se estiman exclusivamente sobre el train de cada split.
- Las colas y grupos empleados en pruebas predictivas se definen sobre train.
- La exploración de lags se restringe al bloque de desarrollo; el holdout no se usa para elegir hipótesis.

Este último punto hace al pipeline más conservador que la exploración inicial del chat y puede producir pequeñas diferencias numéricas.

## 7. Familias de variables

### Topología transaccional

- nivel `BTC_y`;
- primera diferencia, cambio absoluto y diferencia logarítmica;
- sorpresa de Shannon `-log(BTC_y)` y logit;
- lags entre 10 minutos y 6 horas;
- medias, dispersiones, momentum y desvíos respecto de medias/EMA.

### Estado de volatilidad

- volatilidad RMS pasada;
- lags multiescala;
- medias, dispersiones y desvíos de volatilidad.

### Interacciones

Productos entre el estado de volatilidad observable y niveles, cambios, dispersión o desvíos de `BTC_y`. Estas variables representan la hipótesis condicional:

\[
P(Z_{future}\mid\sigma_{past},BTC_y)
\neq
P(Z_{future}\mid\sigma_{past}).
\]

Las variables siguen siendo **componentes NPP de primer orden**: se obtienen de precios y del simplex, son transferibles y auditables, pero no incorporan todavía microestructura, order book, costos efectivos, volatilidad condicional avanzada ni mediciones directas y separadamente identificadas de `gp` y `η`.

## 8. Lectura de la salida

La métrica primaria de clasificación es Macro-F1 porque da el mismo peso a cada régimen. Accuracy puede subir aunque el modelo ignore por completo una cola minoritaria; por eso debe leerse junto con balanced accuracy, recall por clase y log-loss.

Una conclusión favorable requiere simultáneamente:

- ganancia incremental positiva frente al control de igual arquitectura;
- estabilidad en la mayoría de los folds cronológicos;
- recall no nulo y económicamente útil en colas;
- ausencia de una degradación desproporcionada en log-loss;
- coherencia entre resultados holdout y walk-forward.

La importancia de features no se interpreta como causalidad ni como porcentaje de explicación. Las variables son correlacionadas y pueden repartirse o sustituirse importancia entre sí.

## 9. Uso posterior como gate

La salida candidata es:

\[
[P_{XL},P_L,P_M,P_H,P_{XH}],
\]

que puede ingresar al ensamble/GAT o ponderar especialistas:

\[
\hat y=\sum_{k=1}^{5}P(Z=k\mid X_t)f_k(X_t).
\]

Este script no conecta todavía el gate al GAT: primero establece si el vector de régimen posee valor fuera de muestra y si ese valor proviene realmente de la coordenada informacional `BTC_y`.

## 10. Reproducción

En Google Colab:

```bash
pip install -r requirements_topologia_informacional.txt
python analisis_topologia_informacional_btc.py --mount-drive
```

Localmente:

```bash
python analisis_topologia_informacional_btc.py \
  --price-path /ruta/0626p.parquet \
  --simplex-path /ruta/0626dfyp.parquet \
  --output-dir resultados/topologia_informacional_btc
```

Para una prueba de integración sin los Parquet:

```bash
python analisis_topologia_informacional_btc.py \
  --synthetic --synthetic-rows 4000 --folds 2 \
  --backend histgb --output-dir resultados/smoke_topologia
```

