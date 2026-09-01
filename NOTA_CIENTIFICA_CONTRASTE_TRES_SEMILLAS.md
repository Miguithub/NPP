# Nota científica: contraste inicial de tres semillas

## Diseño

Se ejecutaron tres semillas emparejadas (`11`, `23` y `42`) sobre el mismo conjunto temporal de test de 2.016 observaciones. Permanecieron fijos los datos, features, cortes cronológicos, purga, escalado de train, grafo causal, hiperparámetros y criterio de early stopping. Cambiaron únicamente la inicialización y las fuentes aleatorias del entrenamiento.

Las salidas comparadas fueron:

1. persistencia: `p(t)` como predicción de `p(t+1)`;
2. GAT libre: cabeza de logits entrenada solo con entropía cruzada;
3. GAT + NPP: misma cabeza predictiva con regularización NPP;
4. cabeza energética NPP: `softmax(-gp/eta)` producida por el modelo GAT + NPP.

## Resultados medios

| Modelo | CE | KL | JS | MAE | RMSE | Aitchison | Aitchison activos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistencia | 3.056244 | 0.263938 | 0.051146 | 0.000969 | 0.008658 | 2.777890 | 2.872151 |
| GAT libre | 2.974464 | 0.182158 | **0.038953** | 0.00084247 | 0.00727189 | 4.600872 | 3.307978 |
| **GAT + NPP** | **2.974079** | **0.181773** | 0.038959 | **0.00084213** | **0.00726282** | 4.648013 | 3.309358 |
| Cabeza energética NPP | 2.974332 | 0.182026 | 0.039079 | 0.00084416 | 0.00727752 | **4.616565** | **3.307269** |

Frente a persistencia, GAT + NPP redujo aproximadamente 31,1% el KL, 23,8% el JS, 13,1% el MAE y 16,1% el RMSE.

## GAT + NPP frente a GAT libre

GAT + NPP ganó CE y KL en dos de tres semillas. Ganó MAE y RMSE en las tres. La mejora media fue pequeña: aproximadamente `-0.000385` en CE/KL y `-0.00000907` en RMSE. JS quedó en empate práctico.

El intervalo bootstrap de CE/KL todavía atraviesa levemente cero. Con tres semillas no corresponde afirmar significancia ni robustez; este bloque constituye un control operativo y una señal preliminar.

## Cabeza energética

La cabeza energética perdió frente a la salida principal GAT + NPP en CE, KL, JS, MAE y RMSE en las tres semillas, pero ganó Aitchison completo en las tres. Esto revela un intercambio consistente: la salida libre regularizada asigna mejor la masa observada, mientras la parametrización energética reproduce mejor la geometría relativa de log-ratios.

Al restringir Aitchison a nodos activos, la diferencia entre arquitecturas se reduce fuertemente. Esto confirma que nodos nulos o diminutos explican una parte sustancial de la divergencia composicional.

## Limitación crítica del control denominado “GAT libre”

El GAT libre actual no es completamente neutral respecto del NPP. Comparte el mismo grafo híbrido que GAT + NPP:

\[
A_{ij,t}=0.75\,S^{CLR}_{ij,t}+0.25\,S^{E}_{ij,t},
\]

donde la proximidad energética se calcula causalmente mediante `E(t)=-CLR(p(t))`. Por tanto, ambos modelos reciben información energética en la topología antes de diferenciarse por la loss.

La comparación actual identifica el aporte marginal de la regularización NPP sobre un encoder ya informado por energía; no identifica el aporte total del principio frente a un GAT neutral.

## Experimento de control requerido

La siguiente etapa debe añadir un GAT puro con `edge_gamma_npp=0` y `edge_alpha_clr=1`. El diseño completo será:

1. GAT puro con aristas CLR;
2. GAT con grafo híbrido, sin loss NPP;
3. GAT con grafo híbrido y regularización NPP;
4. cabeza energética acoplada;
5. posteriormente, modelo energético autónomo.

Este diseño separará el efecto de la energía incorporada en las aristas del efecto de la restricción energética incorporada en la función objetivo.

## Conclusión provisional

Las tres semillas muestran que los GAT superan ampliamente a persistencia. GAT + NPP presenta una ventaja predictiva pequeña pero consistente en MAE y RMSE frente al GAT libre actual. Sin embargo, la similitud entre ambos puede explicarse parcialmente por el grafo energético compartido. La evidencia favorece continuar el contraste, pero no permite todavía atribuir causalmente la mejora a la regularización NPP ni afirmar superioridad general.

