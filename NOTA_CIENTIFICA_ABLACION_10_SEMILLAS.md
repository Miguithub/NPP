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

## Auditorías complementarias: secciones 15 a 18

Las secciones 15 a 18 amplían la ablación principal con cuatro preguntas distintas: intensidad del mecanismo, identificación interna, coherencia composicional y sensibilidad a la masa del simplex. Las secciones 15, 16 y 17 se mantienen dentro de validación y utilizan tres semillas; la sección 18 reutiliza las predicciones de test ya congeladas de las diez semillas, sin reentrenar ni seleccionar hiperparámetros con test.

### Sección 15: intensidad del mecanismo NPP

La grilla combina cuatro valores de `edge_gamma_npp` (`0`, `0.10`, `0.25`, `0.50`) con cuatro escalas comunes de los términos NPP de la loss (`0`, `0.5`, `1`, `2`). Se completaron 48 corridas: 16 configuraciones por tres semillas emparejadas.

La mejor configuración según CE media de validación fue:

| `edge_gamma_npp` | `lambda_scale` | CE | Aitchison | CLR-MSE |
|---:|---:|---:|---:|---:|
| 0.50 | 1.00 | **2.961361** | 4.662777 | 21.800400 |

El control sin NPP (`gamma=0`, `lambda_scale=0`) obtuvo CE `2.963339`, Aitchison `4.515624` y CLR-MSE `20.456305`. La configuración seleccionada redujo CE aproximadamente `0.0668%` y ganó al control en las tres semillas. Al promediar todas las escalas de loss, la CE evolucionó de forma casi monótona con la intensidad energética de las aristas:

| `edge_gamma_npp` | CE media marginal |
|---:|---:|
| 0.00 | 2.962748 |
| 0.10 | 2.962581 |
| 0.25 | 2.962124 |
| 0.50 | **2.961767** |

Esto aporta evidencia preliminar de una relación dosis-respuesta en la topología. Sin embargo, `gamma=0.50` es el límite superior estudiado: no se identificó todavía un máximo interior y no corresponde afirmar que `0.50` sea el óptimo definitivo.

La respuesta marginal a `lambda_scale` fue no lineal:

| `lambda_scale` | CE media marginal |
|---:|---:|
| 0.0 | 2.962938 |
| 0.5 | 2.962132 |
| 1.0 | **2.961938** |
| 2.0 | 2.962212 |

La región favorable se ubica aproximadamente entre `0.5` y `1.0`, pero el mejor valor depende de `gamma`; existe interacción entre topología y regularización. Tampoco hay una mejora gratuita: frente al control, la configuración con mejor CE empeora Aitchison cerca de `3.26%` y CLR-MSE cerca de `6.57%`. Los términos NPP de esta grilla favorecen el ajuste de masa medido por CE, pero no imponen por sí mismos una solución adecuada en log-ratios.

La conclusión es exploratoria. Treinta y cuatro de las 48 corridas seleccionaron la época máxima `20`, por lo que parte de la superficie puede no haber convergido completamente. La configuración debe confirmarse con un límite de épocas mayor, más semillas y ventanas temporales externas.

### Sección 16: identificación interna de los términos NPP

Manteniendo fijo el grafo híbrido, la CE energética aislada obtuvo la mejor CE de la cabeza predictiva:

| Variante | Cabeza evaluada | CE | Aitchison | CLR-MSE |
|---|---|---:|---:|---:|
| CE energética | `p` | **2.961651** | 4.571888 | 20.963333 |
| Residuo NPP | `p` | 2.961745 | 4.641931 | 21.607565 |
| Residuo + CE energética | `p` | 2.961799 | 4.628488 | 21.483126 |
| NPP completo | `p` | 2.961806 | 4.635802 | 21.550880 |
| Predictiva sola | `p` | 2.962534 | 4.535048 | 20.629574 |
| Energía autónoma | `q_npp` | 2.969290 | **3.896464** | **15.280940** |

La CE energética mejoró la CE predictiva alrededor de `0.030%` y ganó en las tres semillas. La combinación de términos no superó su promedio, de modo que no aparece una sinergia aditiva simple con los pesos actuales.

La cabeza `q_npp=softmax(-gp/eta)` quedó solo `0.228%` por detrás de la cabeza libre en CE, mientras redujo Aitchison aproximadamente `14.1%` y CLR-MSE `25.9%`. Es evidencia favorable para la parametrización energética como sesgo inductivo. No debe interpretarse todavía como identificación causal autónoma del principio: `q_npp` conserva las mismas entradas, el mismo GAT, el grafo híbrido y supervisión mediante su propia CE, y las cabezas latentes `gp` y `eta` mantienen flexibilidad funcional.

### Sección 17: coherencia composicional

`edge_gamma_npp` y `lambda_geometry` operan en lugares distintos. El primero modifica la topología local —quién intercambia información con quién—; el segundo penaliza directamente la geometría global CLR de la predicción:

`L_total = L_pred + lambda_geometry * L_geometry`.

El sweep produjo:

| `lambda_geometry` | CE | Aitchison | CLR-MSE |
|---:|---:|---:|---:|
| 0 | **2.961797** | 4.633726 | 21.531628 |
| 1e-4 | 2.961818 | 4.263099 | 18.260522 |
| 1e-3 | 2.964038 | 2.133120 | 4.981605 |
| 1e-2 | 2.967488 | **2.036631** | **4.585016** |

`lambda_geometry=1e-3` constituye un punto de compromiso preliminar: incrementa CE solo `0.076%`, reduce Aitchison cerca de `54%` y reduce CLR-MSE cerca de `77%`. El deterioro Aitchison observado en los GAT principales no es una incapacidad de la arquitectura; es, en parte, consecuencia de optimizar una función que prioriza la masa económica y no la geometría log-ratio.

La aparente compatibilidad entre objetivos no demuestra por sí sola alineación de gradientes. Deben registrarse el coseno entre `grad(L_pred)` y `grad(L_geometry)`, sus normas relativas, la frecuencia de conflicto y su evolución por capa y época. Una CE casi constante también podría surgir de una región predictiva plana, una contribución geométrica pequeña después del escalado o una separación parcial de parámetros.

### Sección 18: sensibilidad a la masa del simplex

La mejora CE/KL de GAT + NPP frente al GAT libre persiste después de retirar BTC, `U`, ambos, los cinco nodos principales y los diez principales. La mejora relativa en KL pasa de aproximadamente `0.71%` en el simplex completo a cerca de `1.10%` sin el top-10. La evidencia NPP no depende exclusivamente de acertar sobre BTC o `U`.

La auditoría es pospredictiva: se retiran componentes y se vuelve a cerrar el simplex después de predecir, sin reentrenar. Por tanto, demuestra robustez aritmética de la ventaja, pero no ausencia de influencia indirecta de esos nodos durante el entrenamiento del grafo.

El error cuadrático del GAT + NPP está fuertemente concentrado:

- `U`: aproximadamente `50.89%`;
- BTC: aproximadamente `23.27%`;
- ambos: aproximadamente `74.16%`;
- top-10: aproximadamente `92.47%`.

Esto explica por qué RMSE queda dominado por pocos componentes mientras MAE representa mejor al nodo típico. La identidad económica de `U` sigue requiriendo verificación contra la fuente.

Al restringir el cálculo a nodos activos, Aitchison del GAT + NPP cae de `4.6576` a `3.3165`, una reducción aproximada de `28.8%`. Los ceros estructurales y las masas diminutas explican una parte importante de la divergencia: el softmax genera valores positivos mientras el target puede contener ceros, y el log-ratio amplifica esas diferencias. No explican todo el fenómeno, porque persistencia conserva Aitchison `2.8722` entre activos.

CE y KL tampoco constituyen dos evidencias independientes dentro de un mismo escenario: `CE = H(target) + KL`, y la entropía del target es común a todos los modelos comparados.

## Componentes NPP de primer orden y alcance de mejora

La implementación actual utiliza **componentes NPP de primer orden**. En esta nota, *primer orden* designa una primera operacionalización empírica y reducida del principio, no necesariamente una expansión perturbativa formal. Sus elementos centrales son:

- proximidad energética de primer orden en el grafo, aproximada a partir de la composición CLR histórica;
- energía latente `E=gp/eta`, con `gp` y `eta` aprendidos por cabezas neuronales;
- residuo de cierre NPP, CE energética, anclaje de escala y suavidad temporal;
- variables observables de primera generación derivadas principalmente de OHLCV y de las cuotas del simplex.

Estos componentes son deliberadamente iniciales y están sujetos a mejora. No miden todavía de forma directa resistencia, fricción, disipación o restricciones sistémicas. Una operacionalización de orden superior puede incorporar volatilidad realizada y asimétrica, impacto de mercado, spread, profundidad, imbalance, slippage, congestión, métricas on-chain y restricciones institucionales. También puede imponer mayor identificabilidad sobre `gp` y `eta`, en lugar de permitir que ambos sean absorbidos por cabezas latentes flexibles.

La conclusión empírica debe limitarse a esta formulación concreta: existe evidencia favorable para proxies NPP de primer orden, pero no se afirma que representen la forma final u óptima del principio. Mejorar su medición puede amplificar, reducir o redistribuir el efecto observado.

## Síntesis científica acumulada

1. **El GAT produce la mayor parte de la ganancia.** Los tres GAT superaron a persistencia en CE, KL, JS, MAE y RMSE en las diez semillas. Frente a persistencia, GAT + NPP redujo aproximadamente `31.1%` KL, `23.8%` JS, `13.0%` MAE y `16.0%` RMSE.
2. **El aporte NPP es pequeño pero reproducible.** Frente al GAT libre, GAT + NPP ganó CE, KL y JS en `10/10` semillas, MAE en `9/10` y RMSE en `7/10`. Los intervalos emparejados respaldan CE, KL, JS y MAE, pero no resuelven RMSE.
3. **La evidencia NPP más limpia aparece en la topología.** NPP solo aristas ganó al GAT libre en CE, KL, JS y MAE en `10/10` semillas. El contenido energético de las relaciones entre nodos parece más estable que la regularización completa.
4. **La loss NPP adicional cambia el tipo de error.** Frente a solo aristas, el modelo completo mejora CE/KL en `9/10`, pero pierde en promedio JS, MAE y Aitchison. No existe un modelo universalmente superior para todas las geometrías.
5. **La intensidad no es irrelevante.** La sección 15 muestra una región favorable de pesos y una respuesta topológica casi monótona dentro del rango estudiado. El mejor `gamma` quedó en el borde y requiere extensión confirmatoria.
6. **La CE energética aporta señal auxiliar.** Su ablación mejora ligeramente la cabeza `p`, mientras `q_npp` conserva casi toda la CE y mejora sustancialmente la geometría de validación.
7. **La incoherencia Aitchison puede corregirse.** Una penalización CLR explícita reduce radicalmente el error log-ratio con un sacrificio pequeño de CE; el conflicto no es inevitable.
8. **La evidencia no está concentrada únicamente en BTC o `U`.** La ventaja CE/KL persiste al retirar nodos dominantes, aunque RMSE sí está dominado por un conjunto pequeño.
9. **Los ceros estructurales importan, pero no explican todo.** Condicionar a nodos activos reduce Aitchison de los GAT, sin eliminar la ventaja de persistencia en esa métrica.
10. **Las semillas no sustituyen validación temporal.** Las diez semillas prueban estabilidad de optimización sobre un corte económico único; las auditorías 15–17 son además exploratorias y se realizaron con tres semillas de validación.

En conjunto, los resultados respaldan al NPP como un **sesgo inductivo predictivo y estructural compatible con los datos**, especialmente a través de la topología energética. No constituyen todavía una demostración causal general del principio.

## Agenda experimental depurada

La grilla inicial de intensidad, las ablaciones internas, la cabeza energética, la loss CLR y la sensibilidad a nodos dominantes ya fueron ejecutadas. La agenda pendiente excluye esas pruebas y se concentra en extensiones nuevas:

| Familia | Experimentos nuevos | Qué busca | Cómo ayuda al modelo teórico |
|---|---|---|---|
| Robustez temporal externa | Walk-forward con ventanas móviles y expansivas | Ver si la ventaja se repite en otros períodos | Distingue estabilidad temporal de una coyuntura particular |
| Condiciones de validez | Regímenes de volatilidad, liquidez, concentración y estrés | Identificar cuándo NPP aporta más o menos | Permite formular condiciones de frontera del principio |
| Confirmación estadística temporal | Moving-block bootstrap, Diebold–Mariano y Model Confidence Set | Medir incertidumbre respetando autocorrelación | Complementa las semillas con inferencia sobre observaciones económicas dependientes |
| Refinamiento del óptimo de frontera | Extender `gamma_npp` a `0.60`, `0.75` y `1.00`; refinar `lambda_scale` entre `0.5` y `1.0` | Encontrar un máximo interior | Determina la intensidad efectiva, porque el mejor `gamma` actual quedó en el borde |
| Respuesta desacoplada | Variar por separado CE energética, residuo, anclaje y suavidad | Evitar que un multiplicador común mezcle mecanismos | Produce una dosis específica para cada término NPP |
| Compatibilidad de gradientes | Coseno, normas relativas y análisis por capa | Medir cooperación o conflicto entre CE y CLR | Contrasta directamente la compatibilidad geométrica de los objetivos |
| Falsificación topológica | Energías permutadas, grafo aleatorio de igual densidad y rewiring con grados conservados | Ver si importa el contenido NPP de las aristas | Separa señal económica de una regularización genérica del grafo |
| Identificabilidad energética | Cabeza energética libre equivalente; proxies observables o restricciones sobre `gp` y `eta` | Distinguir NPP de una cabeza auxiliar flexible | Aísla el contenido económico de la parametrización |
| Simplex con ceros | Modelo de actividad más distribución ILR condicional; sensibilidad a pseudocounts | Tratar ceros estructurales explícitamente | Evita penalizaciones logarítmicas artificiales sobre nodos inactivos |
| Universo y denominador dinámicos | Altas/bajas cronológicas, distintos `YS_p`, universos sin stablecoins | Medir sensibilidad al cierre del simplex | Comprueba invariancia al universo económico y al denominador |
| Representación empírica de segundo orden | Parkinson, Garman–Klass, Rogers–Satchell, semivarianzas, saltos, Amihud y turnover | Mejorar proxies observables de resistencia y disipación | Acerca las variables a los conceptos matemáticos del NPP |
| Microestructura directa | Spread, profundidad, imbalance, slippage e impacto | Medir fricción en lugar de aproximarla | Fortalece la correspondencia entre teoría y observables |
| Baselines composicionales | VAR en ILR, estado-espacio logistic-normal y regresión Dirichlet | Superar controles más exigentes que persistencia | Separa el aporte NPP del beneficio general de modelar el simplex |
| Baselines neuronales sin grafo | MLP temporal, TCN o Transformer con capacidad comparable | Medir el valor añadido específico de las relaciones | Distingue estructura sistémica de capacidad no lineal genérica |
| Horizontes predictivos | `t+1`, `t+5`, `t+20` y acumulados | Comparar dinámica rápida y estructura lenta | Contrasta si `p` domina en corto plazo y `q_npp` en horizontes mayores |
| Transferencia externa | Otro exchange, período o familia de activos | Medir generalización fuera del ecosistema original | Es la prueba más fuerte de transferibilidad del principio |

Todo ajuste futuro debe seleccionarse exclusivamente con train/validación. El test actual permanece congelado; la confirmación final debe realizarse en ventanas temporales nuevas y nunca utilizadas para elegir pesos, variables o arquitectura.
