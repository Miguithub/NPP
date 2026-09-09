# Fin de ciclo de estos datos: GAT y NPP

Cierre consolidado: 9 de septiembre de 2026. Continuación del contraste y de la ablación de diez semillas, ya integrados en `main`. Las cifras describen ejecuciones anteriores; esta actualización es documental y no genera nuevos resultados.

Este documento cierra el ciclo exploratorio sobre el conjunto de datos y las predicciones disponibles; no declara agotado el potencial de los datos, de la arquitectura ni del principio NPP. Las notas anteriores se conservan como historia del experimento y deben leerse junto con estas salvedades posteriores.

## Alcance y procedencia

Se preservan las 67 celdas anteriores a este cierre, incluidos código, salidas, metadatos y notas. Se amplía únicamente la sección 25 existente. La evidencia procede de las salidas y notas del notebook, de los resúmenes de la ablación de diez semillas y de las tablas guardadas de desarrollo. Las cifras de una ejecución de referencia, las medias entre semillas y los resúmenes por nodo no se intercambian como si fueran la misma agregación.

El notebook estudia 471 nodos, con barras de 10 minutos y horizonte de una barra; el test de referencia contiene 2.016 instantes. No debe confundirse con el nuevo pipeline de extracción de cinco minutos. Se pronostican participaciones transaccionales, no rendimientos.

En GitHub se incluyen el notebook, este cierre en Markdown, tres tablas resumen de desarrollo y un registro de verificación. Las tablas históricas de tres y diez semillas ya están en el repositorio. Los restantes artefactos externos, pesos y datos de entrada continúan en las rutas de Drive documentadas por el notebook; esta publicación no afirma copiarlos todos.

Fuente: [GAT crypto - contraste experimental](https://drive.google.com/file/d/1oQS_XZ57mKxqUtMb4JZSMWrQIvu24PNX/view).

## Balance de la predicción continua

En el test cronológico original de diez semillas, GAT + NPP reduce aproximadamente CE 2,68%, KL 31,05%, JS 23,80%, MAE 12,98% y RMSE 16,02% frente a persistencia. Frente al GAT libre, las reducciones son mucho menores: KL 0,71%, MAE 0,18% y RMSE 0,04%. NPP solo en aristas obtiene mejores medias de JS y MAE que la regularización completa; no existe dominio de una variante en todas las métricas.

CE y KL no son evidencia independiente: para un mismo objetivo difieren por su entropía. La ventaja continua no implica rentabilidad, anticipación de precios ni identificación causal de gp y eta. Las variables constituyen proxies iniciales derivados de OHLCV y del simplex, no mediciones estructurales directas del mecanismo teórico.

## Arquitectura, índice y alcance de la identificación

El GAT procesa el vector completo de participaciones: cada activo es un nodo con historia y factores propios; la atención combina mensajes de los vecinos y el softmax final normaliza entre activos. El GAT libre puede aprender relaciones sin imponer el canal energético NPP. En la ablación de diez semillas, el control **100% libre** usa similitud CLR pura; no debe confundirse con cualquier modelo anterior llamado simplemente «libre».

El peso `gamma_npp` modifica la combinación de similitud histórica CLR y proximidad energética usada para seleccionar y ponderar aristas. No ordena directamente aumentar o disminuir la cuota de un nodo. Los pesos de las aristas entran como sesgo logarítmico en la atención; el aprendizaje determina cómo usar esos mensajes. `lambda_scale` modifica el peso de los términos auxiliares NPP durante el entrenamiento, mientras `lambda_geometry` penaliza discrepancias CLR de la distribución predicha. Son intervenciones diferentes.

El notebook incorpora como feature el índice operativo `n = EMA_precio_1m / sorpresa`, con sorpresa de Shannon, además de rezagos y otros factores. Este contraste no aísla por sí solo el aporte individual de esa feature: la ablación principal modifica aristas y pérdida. Por ello, una mejora atribuible al tratamiento NPP no demuestra automáticamente la utilidad marginal del índice de neguentropía tomado por separado.

Las cabezas `gp` y `eta` son latentes. La parametrización energética usa `q_NPP = softmax(-gp/eta)`. Para cuotas positivas, `E = -CLR(p)` representa energías relativas; con ceros requiere el tratamiento numérico documentado. Representar una distribución mediante energías no demuestra el mecanismo económico: la prueba empírica es la utilidad de las restricciones o de la representación para pronosticar fuera del entrenamiento. Las cuotas por sí solas no identifican causalmente `gp` y `eta` por separado.

## Ejecución de referencia y ablación de diez semillas — secciones 10 a 13

En la ejecución de referencia, GAT + NPP superó a persistencia en RMSE en 427 de 440 nodos activos. Frente al GAT libre ganó en RMSE en 246 y en MAE en 163. Son conteos de esa ejecución, no victorias universales ni diez réplicas temporales: la ventaja no se distribuye por igual entre nodos ni métricas.

En la ablación emparejada de diez semillas, los tres GAT superan a persistencia en CE, KL, JS, MAE y RMSE. El canal NPP de aristas mejora CE, KL, JS y MAE frente al control 100% libre en las diez semillas. GAT + NPP obtiene las mejores medias de CE/KL y RMSE, mientras NPP solo aristas obtiene las mejores medias de JS y MAE. El incremento frente al GAT libre es mucho menor que la ganancia total frente a persistencia.

La distancia de Aitchison sigue favoreciendo a persistencia en la ablación original, incluso en la evaluación de activos. Las masas pequeñas y el tratamiento de ceros afectan intensamente los log-ratios; esto exige informar tanto ajuste de masa como geometría. RMSE no es una banda de confianza ni un intervalo simétrico de error. Los intervalos bootstrap sobre deltas entre semillas describen sensibilidad a la aleatoriedad del ajuste condicionada al mismo conjunto temporal; no representan incertidumbre sobre futuros regímenes.

## Intensidad NPP — sección 15

La grilla de 16 configuraciones y tres semillas favorece, dentro de la región evaluada, `gamma_npp=0.50` y `lambda_scale=1.00`: CE de validación 2,961361 frente a 2,963339 del control sin ambos tratamientos, una reducción aproximada del 0,0668%. La CE media mejora al aumentar el componente de aristas en el intervalo explorado; la respuesta a la regularización es no lineal, con una región favorable entre 0,5 y 1 según las notas de ejecución.

Esto es una señal exploratoria de respuesta a la intensidad, no una ley monótona ni un óptimo universal. `gamma=0.50` está en el borde de la grilla. La mejor CE empeora Aitchison aproximadamente 3,26% y CLR-MSE 6,57% frente al control. Las notas registran que 34 de 48 corridas eligieron la época máxima 20; ese límite impide asumir convergencia suficiente o saturación del modelo.

## Identificación interna — sección 16

Con el grafo híbrido fijo, `energy_ce` aislada obtiene CE de validación 2,961651 frente a 2,962534 de la pérdida predictiva sola. El residuo también aporta una pequeña mejora, pero sumar componentes no produce una mejora aditiva garantizada. Estas comparaciones aíslan tratamientos del modelo, no causas económicas.

La cabeza energética autónoma alcanza CE 2,969290: alrededor de 0,23% peor que la cabeza predictiva de control con el mismo grafo, a cambio de reducir Aitchison aproximadamente 14% y CLR-MSE 26%. Es una alternativa con un compromiso diferente entre masa y geometría. No prueba que las variables latentes sean mediciones de capitales reales.

## Coherencia composicional — sección 17

| `lambda_geometry` | CE de validación | Aitchison | CLR-MSE |
|---|---:|---:|---:|
| 0 | 2,961797 | 4,633726 | 21,531628 |
| 0,001 | 2,964038 | 2,133120 | 4,981605 |

En este sweep, `lambda_geometry=0.001` reduce Aitchison cerca del 54% y CLR-MSE cerca del 77%, con un aumento de CE de aproximadamente 0,076%. Es un compromiso de validación prometedor, condicionado al criterio elegido; no una superioridad única sobre toda la frontera de Pareto. Aumentar a 0,01 mejora algo más la geometría a costa de mayor CE.

La hipótesis de gradientes compatibles entre pérdida predictiva y geométrica sigue pendiente: un sweep de métricas finales no identifica el signo del producto interno de sus gradientes. Haría falta medirlo en parámetros compartidos durante el entrenamiento. Tampoco corresponde atribuir esta mejora geométrica a las predicciones congeladas de modelos entrenados sin esa penalización.

## Sensibilidad de nodos y cola — secciones 18 y 19

`U` y BTC concentran aproximadamente 74% del error cuadrático del GAT + NPP; los diez mayores concentran alrededor de 92,5%. La identidad económica de `U` debe verificarse contra las columnas fuente. La auditoría conserva una ventaja CE/KL frente al libre al excluir nodos dominantes y volver a cerrar el simplex, de modo que esa ventaja no depende exclusivamente de ellos.

Excluir componentes y renormalizar cambia el objeto evaluado: es sensibilidad de las predicciones existentes, no reentrenamiento ni intervención causal. La cola puede tener MAE diminuto y errores CLR elevados. Tampoco debe equipararse automáticamente un cero observado a un cero estructural económico; hay que distinguir inactividad, ausencia y reglas de construcción del dato.

## Estabilidad dentro del test — sección 20

Según el resumen diario registrado, los tres GAT superan a persistencia en CE/KL, JS, MAE y RMSE en los 14 bloques diarios. NPP solo aristas supera al libre los 14 días en CE/KL, JS y MAE, pero solo 7 de 14 en RMSE. NPP completo gana al libre 14 de 14 en CE/KL, 12 de 14 en JS y MAE, y 8 de 14 en RMSE. El deterioro geométrico de la ablación original también persiste en este análisis.

Hay estabilidad descriptiva dentro del período observado, particularmente para el canal de aristas. No es un walk-forward con reentrenamiento: los pesos están congelados y los bloques pertenecen al mismo test. Los acumulados se solapan y los bloques diarios pueden conservar dependencia temporal.

## Bins de nivel y transiciones — secciones 21 y 21.b

Discretizar participaciones continuas cambia el objetivo: acertar un valor cercano no garantiza acertar su intervalo. Persistencia domina el acierto exacto de niveles en varias comparaciones. Un accuracy elevado puede reflejar permanencias frecuentes; por eso se separan permanencia, cambio y destino del cambio.

En la cohorte de 447 nodos variables en train, el resumen agrupado de detección de cambios de bin es:

| Modelo | Recall de cambio | Precisión | F1 | Falsas alarmas entre permanencias |
|---|---:|---:|---:|---:|
| GAT libre | 32,08% | 52,27% | 39,75% | 6,38% |
| NPP solo aristas | 33,32% | 55,47% | 41,63% | 5,82% |
| GAT + NPP | 32,77% | 52,63% | 40,38% | 6,42% |

NPP de aristas ofrece el mejor equilibrio en esa comparación, pero detecta solo un tercio de los cambios. Detectar un cambio no equivale a acertar su destino. Persistencia acierta permanencias y falla cambios por construcción; recuperar cambios puede no compensar las falsas alarmas en accuracy total.

En los nodos constantes en train aparecen muchas falsas alertas al cruzar un umbral diminuto. Se deben conservar como cohorte separada. Las medias macro por nodo y las métricas agrupadas por observaciones tienen denominadores distintos y no se sustituyen entre sí.

## Pérdida, estabilidad y ganancia — sección 22

Se etiqueta `delta = p(t+1)-p(t)` con `tau_j=max(mediana_train(abs(delta_j)), 1e-8)`. La relevancia del cambio es estadística bajo esa regla; el piso numérico no establece relevancia económica.

| Modelo, nodos variables en train | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Persistencia / bin modal | 57,21% | 36,47% | 27,02% |
| GAT libre | 57,19% | 45,11% | 43,60% |
| NPP solo aristas | 57,99% | 45,99% | 44,95% |
| GAT + NPP | 56,71% | 45,30% | 44,21% |

El canal de aristas mejora aproximadamente 1,34 puntos de macro-F1 frente al libre. Su ventaja resulta especialmente visible en pérdidas: recall 39,22%, precisión 79,10% y F1 52,42%, frente a F1 49,24% del libre. En ganancias, la precisión de los tres GAT ronda 28–29%; NPP completo logra el mayor recall, 39,25%, y F1 33,08%, pero las falsas alertas siguen siendo numerosas.

NPP completo detecta más cambios relevantes (recall 51,67%) y también produce más falsas alarmas (29,19%) que NPP de aristas (50,34% y 26,64%). La diferencia de F1 de detección entre ambos es pequeña, alrededor de 0,05 puntos. En nodos constantes en train, un recall casi total convive con falsas alarmas del 100% para cambio relevante: no es buena discriminación. Estas ventajas medias son descriptivas y no acreditan por sí solas significación temporal ni persistencia en cada semilla.

## Sección 23: actividad y cuartiles positivos

Los cortes se obtienen del entrenamiento por activo; se separa la inactividad y se usan cuartiles de valores positivos elegibles. Se discretizan predicciones continuas existentes, sin entrenar un clasificador.

| Modelo | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Bin modal de train | 28,22% | 25,04% | 12,22% |
| GAT libre | 37,56% | 30,10% | 27,06% |
| NPP solo aristas | 37,66% | 29,91% | 26,89% |
| GAT + NPP | 37,07% | 29,72% | 26,40% |
| Persistencia | 47,71% | 38,58% | 38,58% |

Persistencia supera al mejor GAT de cada métrica en aproximadamente 10,04 puntos de accuracy y 11,52 puntos de macro-F1 (calculados antes del redondeo de la tabla). La ventaja no se limita al acierto de la clase dominante. Los GAT superan al bin modal, pero no al último estado observado. NPP completo no mejora esta tarea; el libre tiene las mejores métricas balanceadas entre GAT.

Una predicción continua más cercana puede cruzar incorrectamente una frontera de bin. Esa explicación es compatible con los resultados, no una causa demostrada. Los cuartiles no garantizan balance en test ni en la clase de inactividad. La sección delimita el alcance del método sin demostrar sobreajuste por sí sola.

## Sección 24: actividad y transiciones

Actividad significa participación mayor que el umbral, no presencia económica absoluta. Cambiar el umbral cambia el objetivo. La tabla siguiente corresponde a todos los nodos elegibles; los mejores GAT de accuracy y macro-F1 pueden ser variantes distintas.

| Umbral | Accuracy persistencia | Mejor accuracy GAT | Macro-F1 persistencia | Mejor macro-F1 GAT |
|---|---:|---:|---:|---:|
| 1e-8 | 97,36% | 89,43% | 74,79% | 64,04% |
| 1e-6 | 94,91% | 87,99% | 71,45% | 59,59% |
| 1e-4 | 83,30% | 83,17% | 68,70% | 66,77% |

Persistencia domina globalmente. Con 1e-8, los GAT predicen actividad en todos los casos de la cohorte variable: recall y tasa de falsos positivos de 100%. El accuracy alto refleja prevalencia, no discriminación de inactividad.

No obstante, persistencia no anticipa cambios de estado por construcción. En la cohorte constante en train y umbral 1e-4, GAT + NPP detecta 36,14% de las activaciones, con precisión 39,01% y F1 37,44%; persistencia tiene recall cero. Es una ventaja localizada, no global. No corresponde escoger 1e-4 como umbral óptimo después de mirar test.

## Conclusión del ciclo

Los GAT aportan valor predictivo para la distribución continua de participaciones. NPP aporta una mejora incremental modesta, condicionada por la representación, la pérdida y la tarea, con evidencia favorable al canal topológico en varias comparaciones. Persistencia sigue siendo especialmente fuerte para niveles discretos y estados. No existe un ganador universal.

Diez semillas miden incertidumbre de optimización condicional al mismo dataset, no diez muestras independientes del proceso generador ni bootstrap temporal. Los análisis sucesivos del mismo test son exploratorios. La evidencia no permite afirmar causalidad, validez universal ni proximidad a un límite ontológico de descomposición. La repetición del benchmark determinista por semilla sirve para emparejar comparaciones, no crea diez benchmarks independientes. Evitar leakage de variables no elimina el sesgo de diseñar nuevos análisis después de observar test. Las afirmaciones anteriores de «pre-registro» o «test intacto» se refieren a la etapa original, no al conjunto de lecturas exploratorias posteriores.

## ¿Tenemos evidencia de que la tipología NPP es útil?

Sí, en un sentido predictivo y acotado a este experimento. La comparación entre GAT de base comparable, con y sin representación NPP en las aristas, muestra mejoras pequeñas pero consistentes en varias métricas distributivas del test utilizado. Es defendible concluir:

> La representación relacional NPP contiene estructura útil para predecir ciertos aspectos del sistema, bajo los datos, la arquitectura y el protocolo evaluados.

Aquí «tipología» se refiere a la representación del fenómeno; «topología informacional» es más preciso cuando hablamos de conexiones y pesos del grafo. La utilidad observada puede surgir de una representación o un sesgo inductivo mejor: si NPP se construye a partir de los mismos datos, no implica agregar información externa nueva. Tampoco demuestra causalidad, identificación estructural de gp/eta, ni superioridad universal: los resultados de las secciones 23 y 24 impiden esa generalización.

La idea es conceptualmente afín a [SpotV2Net, Brini y Toscano (2025)](https://doi.org/10.1016/j.ijforecast.2024.11.004), que incorpora estimaciones de volatilidad y covolatilidad en nodos y características de volatilidad de volatilidad en aristas para pronosticar volatilidad intradiaria. La analogía es el uso de información relacional estructurada en un GAT; no se trata de una réplica, ni de evidencia externa que valide específicamente NPP. Sus objetivos, variables y mercados difieren de los de este notebook.

## Siguiente ciclo: tres continuaciones, todavía no ejecutadas

| Continuación | Diseño requerido | Pregunta científica |
|---|---|---|
| Walk-forward con reentrenamiento | Ventanas rolling o expanding; reentrenar cada origen con pasado disponible, validación interna cronológica, purga según horizonte; grafo, escaladores y selección ajustados solo al pasado; evaluar el bloque posterior intacto. | ¿Se sostiene la ventaja cuando cambian el período y el conjunto de entrenamiento? |
| Esquema de regímenes | Definir volatilidad, liquidez, concentración o estrés con información disponible al instante; calibrar cortes en train; comparar modelos dentro de cada régimen, reportando soporte e incertidumbre temporal. Un modelo condicionado por régimen requerirá entrenamiento específico. | ¿Bajo qué restricciones o condiciones de mercado aporta NPP? |
| GAT nativamente entrenado con bins | Definir el objetivo antes del test (nivel, cambio o actividad), estimar cortes solo en train y entrenar una cabeza clasificadora/ordinal con pérdida adecuada. Comparar persistencia y bin modal; reportar macro-F1, balanced accuracy, matrices de confusión, transiciones y calibración. | ¿La desventaja categórica proviene en parte de entrenar para un objetivo continuo distinto? |

Un softmax sobre clases por nodo no garantiza por sí mismo que las cuotas reconstruidas sumen uno entre activos: el nuevo diseño con bins debe explicitar cómo conserva o reconstruye la coherencia composicional. Las clases no deben elegirse por el mejor resultado del test reutilizado.

Estas rutas abren una nueva fase de experimentación; no son resultados de este cierre. Se requiere un protocolo predefinido y bloques de evaluación no usados para seleccionar configuraciones. Se conserva el principio operativo: sin shuffle temporal, sin información futura y con guardado recuperable por corrida.
