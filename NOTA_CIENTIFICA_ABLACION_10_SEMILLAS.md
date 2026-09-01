# Nota científica: ablación NPP con diez semillas

## Pregunta de identificación

El contraste inicial no incluía un control totalmente neutral, porque el denominado GAT libre compartía el grafo híbrido con proximidad energética. Esta ablación separa dos canales distintos del NPP:

1. información energética incorporada en las aristas;
2. regularización NPP incorporada en la función objetivo.

La comparación permite estimar el aporte marginal de cada canal sin cambiar simultáneamente arquitectura, datos o ventana temporal.

## Diseño experimental

Se utilizaron diez semillas emparejadas: `11`, `23`, `42`, `67`, `101`, `137`, `211`, `307`, `401` y `503`. Permanecieron fijos el dataset, las veinte variables por nodo, el horizonte de diez minutos, los cortes cronológicos, la purga, el scaler ajustado solo con train, el top-k, la arquitectura, el optimizador, el early stopping y el test de 2.016 observaciones. No hubo `shuffle`.

Los cuatro modelos fueron:

1. **Persistencia:** predice `p(t+1)=p(t)`.
2. **GAT 100% libre:** loss predictiva y grafo exclusivamente CLR, con `edge_alpha_clr=1` y `edge_gamma_npp=0`. En esta rama del grafo ni siquiera se calcula la proximidad energética.
3. **GAT NPP solo aristas:** loss predictiva sin regularización NPP y grafo híbrido `0.75 CLR + 0.25 energía`.
4. **GAT + NPP:** mismo grafo híbrido y función objetivo regularizada por NPP.

En cada semilla, los tres GAT reiniciaron la aleatoriedad con el mismo valor. Así se comparan inicializaciones equivalentes y el delta es emparejado.

## Resultados medios

| Modelo | CE | KL | JS | MAE | RMSE | Aitchison | Aitchison activos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistencia | 3.056244 | 0.263938 | 0.051146 | 0.00096852 | 0.00865776 | **2.777890** | **2.872151** |
| GAT 100% libre | 2.975592 | 0.183286 | 0.039102 | 0.00084425 | 0.00727389 | **4.588169** | 3.307703 |
| GAT NPP solo aristas | 2.974837 | 0.182531 | **0.038912** | **0.00084184** | 0.00727628 | 4.603160 | **3.304937** |
| GAT + NPP | **2.974289** | **0.181983** | 0.038972 | 0.00084276 | **0.00727083** | 4.657618 | 3.316537 |

Frente a persistencia, GAT + NPP redujo aproximadamente 31,1% el KL, 23,8% el JS, 13,0% el MAE y 16,0% el RMSE. Los tres GAT superaron a persistencia en CE, KL, JS, MAE y RMSE en las diez semillas.

## Canal 1: NPP en las aristas

Frente al GAT 100% libre, el modelo NPP solo aristas ganó en las diez semillas en CE, KL, JS y MAE:

| Métrica | Delta medio: aristas menos libre | IC bootstrap 95% | Victorias |
|---|---:|---:|---:|
| CE / KL | -0.00075448 | [-0.00097249, -0.00051255] | 10/10 |
| JS | -0.00019057 | [-0.00025049, -0.00013150] | 10/10 |
| MAE | -0.00000241 | [-0.00000291, -0.00000198] | 10/10 |
| RMSE | +0.00000239 | [-0.00000216, +0.00000746] | 5/10 |

En el promedio por nodo activo, NPP solo aristas mejoró MAE en 327 de 440 nodos y RMSE en 275. La señal más estable del experimento proviene, por tanto, de la topología energética: mejora la asignación distributiva y el error absoluto sin depender de una semilla particular.

El RMSE agregado no mejora de forma identificable en este contraste. Su intervalo atraviesa cero y está dominado por pocos nodos de gran masa.

## Canal 2: regularización NPP adicional

Frente a NPP solo aristas, el modelo NPP completo ganó CE y KL en nueve de diez semillas:

| Métrica | Delta medio: NPP completo menos aristas | IC bootstrap 95% | Victorias |
|---|---:|---:|---:|
| CE / KL | -0.00054767 | [-0.00088862, -0.00024692] | 9/10 |
| JS | +0.00006084 | [+0.00000663, +0.00011139] | 2/10 |
| MAE | +0.00000092 | [+0.00000012, +0.00000173] | 4/10 |
| RMSE | -0.00000544 | [-0.00001476, +0.00000461] | 7/10 |

La regularización mejora de forma consistente CE/KL, pero presenta un intercambio: el modelo de solo aristas conserva mejor JS y MAE. La mejora de RMSE del modelo completo es pequeña y su intervalo atraviesa cero. Esto es compatible con una regularización que corrige algunos errores grandes o nodos económicamente relevantes sin reducir el error cotidiano de la mayoría de los nodos.

Por nodo activo, GAT + NPP supera al modelo de solo aristas en MAE en 48 de 440 nodos y en RMSE en 175. Por ello no corresponde afirmar dominio universal del modelo completo.

## NPP total frente al GAT neutral

GAT + NPP superó al GAT 100% libre en CE, KL y JS en las diez semillas, en MAE en nueve y en RMSE en siete. Los intervalos bootstrap quedaron completamente bajo cero para CE, KL, JS y MAE; el de RMSE atravesó cero.

La magnitud es pequeña. Frente al GAT libre, el modelo completo mejora aproximadamente 0,71% el KL, 0,33% el JS, 0,18% el MAE y 0,04% el RMSE. En KL, el GAT libre ya explica cerca del 98,4% de la mejora total frente a persistencia y NPP agrega el 1,6% restante.

La interpretación correcta es que NPP constituye un sesgo inductivo pequeño, estable y medible sobre un GAT que produce la mayor parte de la ganancia predictiva.

## Geometría de Aitchison

Persistencia supera ampliamente a todos los GAT en Aitchison. Entre los GAT, el modelo 100% libre obtiene el mejor Aitchison completo y el modelo NPP solo aristas el mejor Aitchison condicionado a nodos activos.

El deterioro permanece al restringir el cálculo a 440 nodos activos. Por tanto, los ceros y masas diminutas explican solo una parte del fenómeno. Los GAT asignan mejor la masa económica según CE, KL, JS, MAE y RMSE, pero reproducen peor ciertos log-ratios, especialmente en la cola. Esto constituye un intercambio real entre ajuste de masa y geometría composicional, no una contradicción algebraica.

## Concentración del error

El RMSE continúa fuertemente concentrado. En GAT + NPP, el nodo `U` explica aproximadamente 50,9% del error cuadrático total y BTC cerca de 23,3%. Los cuatro nodos principales concentran aproximadamente 86%.

La identidad económica de `U` debe auditarse contra la fuente antes de interpretar sustantivamente este resultado. La concentración también explica por qué una mejora en pocos nodos puede mover el RMSE agregado sin producir una mayoría de victorias por nodo.

## Alcance y rusticidad deliberada de las variables

Las veinte variables por nodo constituyen una primera operacionalización relativamente rústica. Son rezagos y transformaciones estadísticas o técnicas de las cuotas, precios y volúmenes disponibles: sorpresa de Shannon, proxies de liquidez y estabilidad, medias móviles, MACD, RSI, skewness y kurtosis, entre otras. En esencia, pueden construirse de manera causal a partir de información OHLCV y de la cuota del activo dentro del volumen total.

Esta simplicidad tiene una ventaja científica: el pipeline es reproducible y transferible a prácticamente cualquier criptoactivo o activo digital con datos de apertura, cierre, máximo, mínimo y volumen. La señal NPP observada no depende de fuentes propietarias, order books completos ni variables on-chain especializadas.

También impone una limitación importante. Las variables actuales son proxies indirectos y no una medición estructural exhaustiva de resistencia o fricción. El experimento no explota todavía, entre otras posibilidades:

- spread bid-ask, profundidad, imbalance y pendiente del libro de órdenes;
- slippage, impacto de mercado, fees, congestión o latencia;
- volatilidad realizada intrabar y estimadores que aprovechen más completamente OHLC;
- modelos de volatilidad condicional y asimétrica como ARCH, GARCH o EGARCH;
- semivarianza, downside risk y otras medidas explícitas de asimetría;
- métricas on-chain, flujos entre exchanges o concentración de tenedores.

Por ello, el resultado actual debe leerse como una prueba conservadora de suficiencia mínima: aun con variables accesibles y poco especializadas, la información NPP produce un efecto pequeño pero estable. No demuestra que esta sea la representación óptima del principio. Variables mejor alineadas con resistencia, disipación, restricciones y fricción podrían amplificar, reducir o redistribuir el efecto.

## Por qué MAE y RMSE tienen escalas distintas

MAE y RMSE no son métricas inversas y ambas conservan la unidad de la cuota. El MAE promedia errores absolutos y trata linealmente cada desviación. El RMSE eleva primero los errores al cuadrado, promedia y luego aplica la raíz; por ello concede mucho más peso a errores grandes y eventos extremos.

En este panel, la masa y el error están concentrados en pocos nodos dominantes. Esa heterogeneidad hace que un conjunto pequeño de errores grandes eleve el RMSE hasta un orden aparente mayor que el MAE, aunque la mayoría de las observaciones tenga errores pequeños. La diferencia entre ambas métricas es información sobre la cola de errores, no una incoherencia de unidades.

## Alcance inferencial

Las diez semillas miden estabilidad respecto de inicialización y optimización sobre un único corte temporal. Los intervalos bootstrap describen esos deltas emparejados; no representan incertidumbre temporal ni diez muestras económicas independientes.

No se utilizó el test para seleccionar hiperparámetros. Este test debe quedar congelado en la siguiente etapa. La robustez externa exige repetir el diseño en ventanas walk-forward nuevas, recalculando scaler y grafo exclusivamente con el pasado de cada ventana.

## Próximo experimento

El paso siguiente es una grilla pequeña, predefinida y seleccionada solo con train/validación para:

- peso energético del grafo;
- intensidad de la regularización NPP;
- ablaciones de familias de variables alineadas con resistencia, fricción, liquidez, concentración e información.

Esto permitirá distinguir si el efecto pequeño se debe a una subponderación del NPP, a variables que aproximan débilmente su matemática o a un aporte estructural verdaderamente modesto. La configuración elegida deberá confirmarse después en nuevas ventanas temporales nunca usadas para ajustarla.
