# Conclusiones consolidadas del GAT y la primera aproximación NPP

Fecha de síntesis: 6 de septiembre de 2026.

Esta nota reúne la comparación de diez semillas y los experimentos de las secciones 15, 16, 17 y 18 de `GAT crypto - contraste experimental`. Los valores corresponden a los resultados examinados en septiembre de 2026. Es una síntesis documental; no representa una nueva ejecución.

## 1. Alcance y criterios de comparación

La comparación principal evalúa cuatro variantes sobre el mismo test cronológico de 2.016 observaciones. Cada nodo representa un activo y la salida conjunta predice sus participaciones en el volumen transaccional total. Las diez semillas son 11, 23, 42, 67, 101, 137, 211, 307, 401 y 503.

- **Persistencia:** proyecta la última participación observada; es determinista.
- **GAT libre:** grafo CLR y objetivo predictivo, sin los componentes NPP evaluados.
- **GAT NPP solo aristas:** incorpora proximidad energética al grafo, conservando el objetivo predictivo.
- **GAT + NPP:** incorpora el grafo híbrido y los objetivos energéticos auxiliares.

Los sweeps de las secciones 15–17 utilizan tres semillas de validación: 11, 23 y 42. La sección 18 reutiliza las predicciones de test de diez semillas. Los valores de validación y test no se comparan directamente para declarar un ganador.

Una mejora porcentual se calcula como `100 * (métrica_referencia - métrica_variante) / métrica_referencia`. Un valor negativo indica deterioro. Todas las métricas siguientes se minimizan.

## 2. Resultados principales fuera de muestra

| Modelo | CE | KL | JS | MAE | RMSE | Aitchison |
|---|---:|---:|---:|---:|---:|---:|
| Persistencia | 3.056244 | 0.263938 | 0.051146 | 0.00096852 | 0.00865776 | 2.777890 |
| GAT libre | 2.975592 | 0.183286 | 0.039102 | 0.00084425 | 0.00727389 | 4.588169 |
| GAT NPP solo aristas | 2.974837 | 0.182531 | 0.038912 | 0.00084184 | 0.00727628 | 4.603160 |
| GAT + NPP | 2.974289 | 0.181983 | 0.038972 | 0.00084276 | 0.00727083 | 4.657618 |

### Reducción porcentual frente a persistencia

| Modelo | CE | KL | JS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| GAT libre | 2.64% | 30.56% | 23.55% | 12.83% | 15.98% |
| GAT NPP solo aristas | 2.66% | 30.84% | 23.92% | 13.08% | 15.96% |
| GAT + NPP | 2.68% | 31.05% | 23.80% | 12.98% | 16.02% |

Los tres GAT superan a persistencia en estas cinco métricas en las diez semillas. La mayor parte de la ganancia aparece al incorporar el GAT. Persistencia conserva el menor Aitchison; sus diez filas corresponden al mismo pronóstico, no a diez realizaciones independientes.

### Aporte incremental frente al GAT libre

| Variante NPP | Mejora CE | Mejora KL | Mejora JS | Mejora MAE | Mejora RMSE |
|---|---:|---:|---:|---:|---:|
| Solo aristas | 0.025% | 0.412% | 0.487% | 0.285% | −0.033% |
| NPP completo | 0.044% | 0.710% | 0.332% | 0.176% | 0.042% |

| Comparación: primera variante frente a segunda | CE/KL | JS | MAE | RMSE |
|---|---:|---:|---:|---:|
| Solo aristas frente a libre | 10/10 | 10/10 | 10/10 | 5/10 |
| NPP completo frente a libre | 10/10 | 10/10 | 9/10 | 7/10 |
| NPP completo frente a solo aristas | 9/10 | 2/10 | 4/10 | 7/10 |

Las fracciones representan semillas ganadas. La evidencia más consistente de las aristas se observa en CE/KL, JS y MAE. La regularización completa añade una mejora CE/KL, con intercambios en otras métricas. Aunque NPP completo presenta el menor RMSE promedio, los intervalos bootstrap entre semillas de sus diferencias frente a los otros GAT incluyen cero: su superioridad en RMSE no queda establecida.

## 3. Interpretación de las métricas

| Métrica | Aspecto evaluado | Resultado principal |
|---|---|---|
| CE/KL | Ajuste distributivo ponderado por la masa real | Favorece al NPP completo |
| JS | Discrepancia simétrica entre distribuciones | Favorece al NPP solo aristas |
| MAE | Magnitud media del error absoluto | Favorece al NPP solo aristas |
| RMSE | Error absoluto con mayor penalización de errores grandes | NPP completo gana en promedio, con incertidumbre entre GAT |
| Aitchison | Error en relaciones logarítmicas entre componentes | Favorece a persistencia; entre GAT, al libre |

Para un target fijo, `CE(p, p_hat) = H(p) + KL(p || p_hat)`. Por ello, CE y KL comparten la misma diferencia absoluta entre modelos. Los porcentajes cambian porque usan distintos denominadores: no constituyen dos evidencias independientes.

MAE y RMSE no son bandas de confianza. El RMSE global del NPP completo equivale aproximadamente a 0.727 puntos porcentuales de participación, pero no describe el error particular de cada activo ni garantiza una banda simétrica de ese ancho.

En el código, la métrica denominada Aitchison es la media temporal de la raíz del error cuadrático medio entre componentes CLR: una versión normalizada por dimensión de la distancia euclídea CLR. Usa un piso numérico para tratar ceros. CLR-MSE y esta distancia resumen errores relacionados, no pruebas geométricas independientes.

## 4. Sección 15: intensidad del mecanismo NPP

La grilla comprende 16 configuraciones y tres semillas: 48 entrenamientos. El mejor punto observado es `gamma_npp = 0.50`, `lambda_scale = 1.0`, con CE media 2.961361 frente a 2.963339 del control `gamma = 0`, `lambda_scale = 0`.

| Métrica | Cambio frente al control |
|---|---:|
| CE | Mejora 0.0668% |
| Aitchison | Empeora 3.26% |
| CLR-MSE | Empeora 6.57% |

La mejora CE aparece en las tres semillas. Promediando las escalas de regularización, CE desciende de 2.962748 a 2.962581, 2.962124 y 2.961767 al aumentar gamma de 0 a 0.10, 0.25 y 0.50. Es evidencia exploratoria de respuesta a la intensidad de la topología dentro de la grilla, no de monotonicidad en cada combinación o de causalidad económica.

Promediando los grafos, las escalas 0, 0.5, 1 y 2 producen CE de 2.962938, 2.962132, 2.961938 y 2.962212. La escala intermedia resulta favorable y aparecen interacciones entre topología y regularización.

El óptimo no es definitivo: gamma 0.50 es el extremo superior y 34/48 corridas seleccionaron la época máxima disponible. Además, `lambda_scale` escala los términos NPP convencionales; no es el peso `lambda_geometry` de la sección 17.

## 5. Sección 16: identificación de componentes energéticos

La mejor CE de esta ablación corresponde a añadir únicamente `energy_ce` a la pérdida predictiva: 2.961651 frente a 2.962534 de la predictiva sola. La mejora es aproximadamente 0.030% y aparece en las tres semillas. La combinación de términos no supera en promedio a este componente aislado con los pesos evaluados.

La cabeza energética autónoma, frente a la predictiva sola, empeora CE 0.228%, mejora Aitchison 14.1% y mejora CLR-MSE 25.9%.

La parametrización energética obtiene un ajuste predictivo cercano y mejor geometría composicional. Su autonomía corresponde a la salida predictiva: sigue utilizando datos, GAT, grafo y supervisión. Estos resultados respaldan su utilidad como representación aprendida, pero no identifican por sí solos un mecanismo económico causal. La flexibilidad de los parámetros latentes gp y eta exige controles adicionales para atribuir la ventaja específicamente al principio.

## 6. Sección 17: coherencia composicional

Con `lambda_geometry = 1e-3`, frente a no incorporar la pérdida CLR, CE empeora 0.076%, Aitchison mejora aproximadamente 54% y CLR-MSE mejora aproximadamente 77%.

Una mejora geométrica grande se consigue con un deterioro pequeño de CE. Esto muestra margen para compatibilizar los objetivos. La hipótesis de compatibilidad entre sus gradientes sigue pendiente de medición directa: los valores finales no determinan el ángulo entre gradientes durante el entrenamiento.

El valor 1e-3 es un compromiso preliminar, condicionado al criterio de selección y al presupuesto de entrenamiento. Doce de las quince corridas seleccionaron la época máxima 20. Las cifras pertenecen a validación con tres semillas y requieren confirmación externa.

## 7. Sección 18: sensibilidad a la masa y a los nodos

La ventaja KL del NPP completo frente al libre permanece al excluir BTC, U, ambos y los principales cinco o diez activos de la evaluación. La mejora relativa pasa de aproximadamente 0.71% en el simplex completo a 1.10% sin los diez principales por participación.

| Nodos | Participación en el error cuadrático del GAT + NPP |
|---|---:|
| U | 50.89% |
| BTC | 23.27% |
| U + BTC | 74.16% |
| Diez mayores contribuyentes al error | 92.47% |

El ranking de contribución al error no es necesariamente el ranking de participación utilizado para excluir el top-10. La exclusión y renormalización es una auditoría de predicciones ya generadas; no equivale a reentrenar sin esos nodos ni demuestra mejora individual en todos ellos. Los grupos definidos usando participaciones observadas en test son descriptivos de ese período; para selección prospectiva deben fijarse con información disponible anteriormente.

Al evaluar solo nodos activos, Aitchison del NPP completo baja de 4.6576 a 3.3165, aproximadamente 28.8%. Los ceros explican una parte importante del problema, pero persistencia sigue siendo mejor entre activos, con 2.8722. Es necesario distinguir ceros estructurales, ausencia de datos y participaciones muy pequeñas antes de una interpretación económica definitiva.

## 8. Alcance teórico y estadístico

Esta primera aproximación al NPP organiza la información disponible mediante una topología y una parametrización energética que aportan mejoras predictivas pequeñas y repetibles. Las ablaciones permiten distinguir contribuciones de aristas, objetivos energéticos y geometría composicional.

La energía de las aristas deriva de −CLR: la distancia entre energías equivale a la diferencia absoluta entre log-participaciones. La evidencia respalda la utilidad de esa organización de los datos; falta contrastarla con medidas independientes de fricción, resistencia y disipación. No implica incorporar información externa adicional.

Las semillas exploran distintas realizaciones aleatorias del aprendizaje sobre una misma trayectoria histórica. El bootstrap de deltas entre semillas cuantifica esa variabilidad de optimización, no la incertidumbre temporal del proceso económico. Los subconjuntos evalúan sensibilidad a la composición del universo. Nuevas ventanas y remuestreo temporal por bloques siguen siendo necesarios.

Los resultados respaldan identificación de componentes computacionales y utilidad predictiva de esta aproximación. No constituyen identificación causal económica ni una prueba concluyente del principio general. La exploración acumulada también exige reservar un bloque temporal nuevo para confirmar las decisiones tomadas a partir de los resultados actuales.

## 9. Expansiones conceptuales pendientes

**Clasificación en bins de 2.5 puntos porcentuales.** Permitiría preguntar en qué rango estará cada participación. Deben evaluarse desbalance de clases, error ordinal, calibración y cierre del simplex reconstruido. Una mayor accuracy puede reflejar una resolución más gruesa. Una primera referencia puede obtenerse discretizando las predicciones ya guardadas, incluida persistencia, antes de comparar con modelos reentrenados para esa tarea.

**Hipótesis de un límite de descomposición.** Todavía no puede afirmarse que se alcanzó un límite ontológico. La mejora geométrica de la sección 17 muestra que parte del desempeño dependía del objetivo de entrenamiento. Quedan abiertas limitaciones de variables, convergencia, arquitectura y régimen temporal. Curvas de aprendizaje, controles de capacidad y análisis de residuos permitirían investigar un techo predictivo; identificarlo con un límite del fenómeno requeriría justificación adicional.

La contribución NPP más consistente está en la topología. La parametrización energética ofrece una representación alternativa útil, y la pérdida CLR abre un margen considerable para mejorar la geometría. El efecto incremental es modesto y su generalización temporal permanece por contrastar.

## Fuentes de resultados

- [Test de diez semillas: resumen por modelo](https://drive.google.com/file/d/1kvKfOnEConOv78oXREp5hVBkDs-mlfzp/view).
- [Test de diez semillas: contrastes emparejados](https://drive.google.com/file/d/1VMK-pwl_j99AlKAOVimgxo7Hu_5xOXaW/view).
- [Opción 3: corridas de validación](https://drive.google.com/file/d/1buroOsOEztin5953pEmLFkQ8jGIk3KZt/view) y [resumen](https://drive.google.com/file/d/1a_vNt_QzLSSGj1D0Y_11QA2XNjUkYqwp/view).
- [Opción 5: identificación](https://drive.google.com/file/d/1f_GgOC7wYmZDF9QSU_Qa-0O5Kd3ehWxc/view).
- [Opción 6: geometría](https://drive.google.com/file/d/182Ryv0_tou6HcbBLOqX6prD8Te7B35cO/view).
- [Opción 7: escenarios](https://drive.google.com/file/d/1_gBNefn0a5nHNh9-YfPtP8VcdAZgAcYW/view) y [estratos de masa](https://drive.google.com/file/d/1a9-8J-Ib8ComQ6eLcPWI02TtVO0J5ly1/view).

Los enlaces apuntan a archivos de Drive que pueden actualizarse. Esta nota conserva los valores analizados; el commit de GitHub fija la versión de la síntesis.
