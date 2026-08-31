# Nota científica: GAT + NPP frente a la cabeza energética NPP

## 1. Alcance de la comparación

`GAT + NPP` y `Cabeza energética NPP` no son dos redes entrenadas de manera independiente. Son dos distribuciones de salida del mismo modelo regularizado por NPP. Ambas reciben las mismas 20 variables por nodo, el mismo grafo causal, las mismas capas de atención y la misma representación latente `h`.

Por lo tanto, esta comparación no prueba todavía que el NPP alcance el mismo resultado con menos datos. Lo que prueba es que una parametrización de salida más restringida alcanza prácticamente el mismo resultado utilizando la misma información interna.

## 2. Salida predictiva principal: GAT + NPP

La cabeza principal transforma directamente la representación latente de cada nodo en un logit:

\[
z_j = w_z^\top h_j+b_z,
\qquad
p_j=\frac{e^{z_j}}{\sum_k e^{z_k}}.
\]

Esta es la salida utilizada como predicción principal. El Softmax garantiza positividad y cierre del simplex. Sus logits poseen libertad para aprender cualquier topografía relativa compatible con la representación del GAT.

Aunque la salida es libre, su entrenamiento no lo es completamente: además de la entropía cruzada predictiva, recibe una penalización que la aproxima a la estructura energética del NPP. En consecuencia, debe interpretarse como una predicción flexible condicionada por el principio.

## 3. Cabeza energética NPP

Desde la misma representación `h`, el modelo estima dos cantidades latentes positivas:

\[
gp_j=\sigma(w_{gp}^\top h_j+b_{gp}),
\qquad
\eta_j=\operatorname{softplus}(w_\eta^\top h_j+b_\eta)+\varepsilon.
\]

Luego define la energía relativa:

\[
E_j=\frac{gp_j}{\eta_j},
\qquad
q_j^{NPP}=\frac{e^{-E_j}}{\sum_k e^{-E_k}}.
\]

Esta salida no puede elegir logits arbitrarios: debe expresar toda la distribución mediante el cociente energético `gp/eta`. Es, por ello, la materialización estricta de la solución de partición propuesta por el NPP.

## 4. Cómo se acoplan ambas salidas

La función objetivo del modelo NPP combina cinco componentes:

\[
\mathcal L=
\mathcal L_{CE}(p,y)
+\lambda_{NPP}\mathcal L_{residual}
+\lambda_E\mathcal L_{CE}(q^{NPP},y)
+\lambda_\eta\mathcal L_{anchor}
+\lambda_S\mathcal L_{smooth}.
\]

- `CE(p,y)` entrena la salida predictiva principal.
- El residuo NPP penaliza la separación entre `log(p)` y la identidad energética `-E-log(Z)`.
- `CE(q_NPP,y)` obliga a que la cabeza energética sea predictiva por sí misma.
- El anclaje de `eta` reduce la indeterminación de escala durante la optimización.
- La suavidad energética penaliza cambios temporales excesivamente bruscos.

Así, la cabeza energética no es un análisis posterior: participa en el entrenamiento y condiciona la geometría aprendida por el modelo completo.

## 5. Resultado empírico de la ejecución de referencia

| Métrica | GAT + NPP | Cabeza energética NPP | Diferencia absoluta |
|---|---:|---:|---:|
| CE | 2.974420 | 2.974582 | 0.000161 |
| KL | 0.182114 | 0.182276 | 0.000161 |
| JS | 0.038999 | 0.039103 | 0.000104 |
| RMSE | 0.007270 | 0.007277 | 0.000006 |
| MAE | 0.000842 | 0.000844 | 0.000001 |
| Aitchison | 4.682563 | 4.660449 | -0.022114 |

Las diferencias son muy pequeñas. La cabeza principal es apenas mejor en CE, KL, JS, MAE y RMSE; la cabeza energética es ligeramente mejor en Aitchison. En términos predictivos agregados, ambas soluciones son casi equivalentes.

## 6. Interpretación científica

La proximidad observada indica que la solución flexible aprendida por los logits se encuentra muy cerca de una distribución representable como `softmax(-gp/eta)`. Esto constituye evidencia de coherencia interna entre el predictor y la restricción NPP.

No obstante, la comparación no demuestra por sí sola que el principio sea causal, único ni informacionalmente mínimo. La similitud puede surgir en parte porque ambas cabezas comparten encoder y porque la loss fue diseñada explícitamente para alinearlas. La evidencia correcta es de suficiencia estructural dentro de esta arquitectura: la restricción energética no destruye capacidad predictiva relevante.

## 7. Identificabilidad

Con cuotas transaccionales solamente se identifica la energía relativa efectiva:

\[
E_j=gp_j/\eta_j,
\]

salvo la invariancia aditiva propia del Softmax. `gp` y `eta` no deben interpretarse separadamente como capital y resistencia económicos observados. Para otorgarles significado causal se requieren proxies externos independientes y restricciones adicionales de identificación.

## 8. Qué experimento falta

Para demostrar compresión informacional o superioridad del principio deben añadirse:

1. un modelo exclusivamente energético, sin cabeza libre;
2. ablaciones de familias de variables;
3. varias semillas de inicialización;
4. múltiples ventanas temporales fuera de muestra;
5. comparación estadística de diferencias por fecha y por nodo;
6. auditoría de nodos dominantes, especialmente el ticker `U`.

Si una arquitectura exclusivamente energética mantiene el desempeño con menos variables o parámetros, entonces sí podrá sostenerse empíricamente que el NPP produce una representación más parsimoniosa y no solamente una salida más restringida.
