# Prompts técnicos por modelo

## 1. regression::persistence

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `persistence` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
Benchmark determinista: proyecta sigma_past como sigma_future. No es una red; no tiene capas, dimensiones, activaciones, Dropout, BatchNorm ni L1/L2.

2. OPTIMIZACIÓN
No se entrena: no hay loss optimizada, optimizador, learning rate, batch size ni épocas.

3. MÉTRICAS
Entrenamiento del ajuste final:
        split      MAE     RMSE        R2  n_train  n_eval
holdout_train 0.001057 0.001702 -0.185763    18215   18215

Holdout cronológico:
  split      MAE     RMSE        R2  n_train  n_eval
holdout 0.000906 0.001411 -0.177636    18215    7808

Train walk-forward, media y desvío:
metric      mean      std
   MAE  0.001047 0.000172
  RMSE  0.001695 0.000270
    R2 -0.122857 0.131952

Validación walk-forward, media y desvío:
metric      mean      std
   MAE  0.001120 0.000256
  RMSE  0.001747 0.000402
    R2 -0.374382 0.244401

Métricas por clase en holdout:
No aplica: el target es continuo.

Matrices de confusión holdout:
No aplica: es un modelo de regresión.

4. CONTEXTO DE LOS DATOS
Problema: regresión de sigma_future a 20 minutos. Features: volatilidad RMS pasada disponible en t. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 2. regression::linear_volatility

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `linear_volatility` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
Regresión lineal OLS de una sola salida, equivalente a una transformación afín con activación identidad. Sin capas ocultas, Dropout, BatchNorm ni regularización explícita.

2. OPTIMIZACIÓN
Mínimos cuadrados ordinarios de scikit-learn; error cuadrático como criterio implícito. No usa learning rate, weight decay, batch size ni épocas iterativas.

3. MÉTRICAS
Entrenamiento del ajuste final:
        split      MAE     RMSE       R2  n_train  n_eval
holdout_train 0.000912 0.001428 0.165757    18215   18215

Holdout cronológico:
  split     MAE     RMSE       R2  n_train  n_eval
holdout 0.00082 0.001191 0.161929    18215    7808

Train walk-forward, media y desvío:
metric     mean      std
   MAE 0.000920 0.000160
  RMSE 0.001440 0.000246
    R2 0.195872 0.059633

Validación walk-forward, media y desvío:
metric     mean      std
   MAE 0.000976 0.000184
  RMSE 0.001491 0.000409
    R2 0.025014 0.079250

Métricas por clase en holdout:
No aplica: el target es continuo.

Matrices de confusión holdout:
No aplica: es un modelo de regresión.

4. CONTEXTO DE LOS DATOS
Problema: regresión de sigma_future a 20 minutos. Features: volatilidad RMS pasada disponible en t. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 3. regression::linear_volatility_plus_topology

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `linear_volatility_plus_topology` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
Regresión lineal OLS de una sola salida, equivalente a una transformación afín con activación identidad. Sin capas ocultas, Dropout, BatchNorm ni regularización explícita.

2. OPTIMIZACIÓN
Mínimos cuadrados ordinarios de scikit-learn; error cuadrático como criterio implícito. No usa learning rate, weight decay, batch size ni épocas iterativas.

3. MÉTRICAS
Entrenamiento del ajuste final:
        split      MAE     RMSE       R2  n_train  n_eval
holdout_train 0.000912 0.001428 0.165773    18215   18215

Holdout cronológico:
  split     MAE     RMSE       R2  n_train  n_eval
holdout 0.00082 0.001191 0.162041    18215    7808

Train walk-forward, media y desvío:
metric     mean      std
   MAE 0.000920 0.000159
  RMSE 0.001439 0.000246
    R2 0.196000 0.059553

Validación walk-forward, media y desvío:
metric     mean      std
   MAE 0.000976 0.000184
  RMSE 0.001491 0.000409
    R2 0.024793 0.079554

Métricas por clase en holdout:
No aplica: el target es continuo.

Matrices de confusión holdout:
No aplica: es un modelo de regresión.

4. CONTEXTO DE LOS DATOS
Problema: regresión de sigma_future a 20 minutos. Features: volatilidad pasada más variación contemporánea de la cuota BTC_y. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 4. regression::histgb_volatility

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `histgb_volatility` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
HistGradientBoostingRegressor: ensamble secuencial de árboles histogramados, max_depth=5 y max_leaf_nodes=15. No usa capas neuronales, activaciones, Dropout o BatchNorm.

2. OPTIMIZACIÓN
Loss squared_error; learning_rate=0.03; max_iter=300; random_state=42. No existe batch size y no se configuró penalización L1/L2 adicional.

3. MÉTRICAS
Entrenamiento del ajuste final:
        split      MAE    RMSE       R2  n_train  n_eval
holdout_train 0.000958 0.00149 0.090774    18215   18215

Holdout cronológico:
  split      MAE    RMSE       R2  n_train  n_eval
holdout 0.000887 0.00125 0.076494    18215    7808

Train walk-forward, media y desvío:
metric     mean      std
   MAE 0.000933 0.000179
  RMSE 0.001454 0.000275
    R2 0.183002 0.095577

Validación walk-forward, media y desvío:
metric      mean      std
   MAE  0.001026 0.000158
  RMSE  0.001531 0.000425
    R2 -0.028953 0.092015

Métricas por clase en holdout:
No aplica: el target es continuo.

Matrices de confusión holdout:
No aplica: es un modelo de regresión.

4. CONTEXTO DE LOS DATOS
Problema: regresión de sigma_future a 20 minutos. Features: volatilidad RMS pasada disponible en t. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 5. regression::histgb_volatility_plus_topology

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `histgb_volatility_plus_topology` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
HistGradientBoostingRegressor: ensamble secuencial de árboles histogramados, max_depth=5 y max_leaf_nodes=15. No usa capas neuronales, activaciones, Dropout o BatchNorm.

2. OPTIMIZACIÓN
Loss squared_error; learning_rate=0.03; max_iter=300; random_state=42. No existe batch size y no se configuró penalización L1/L2 adicional.

3. MÉTRICAS
Entrenamiento del ajuste final:
        split      MAE     RMSE      R2  n_train  n_eval
holdout_train 0.000958 0.001489 0.09265    18215   18215

Holdout cronológico:
  split      MAE    RMSE       R2  n_train  n_eval
holdout 0.000887 0.00125 0.075768    18215    7808

Train walk-forward, media y desvío:
metric     mean      std
   MAE 0.000917 0.000187
  RMSE 0.001419 0.000287
    R2 0.221904 0.125154

Validación walk-forward, media y desvío:
metric      mean      std
   MAE  0.001021 0.000156
  RMSE  0.001531 0.000424
    R2 -0.029403 0.093050

Métricas por clase en holdout:
No aplica: el target es continuo.

Matrices de confusión holdout:
No aplica: es un modelo de regresión.

4. CONTEXTO DE LOS DATOS
Problema: regresión de sigma_future a 20 minutos. Features: volatilidad pasada más variación contemporánea de la cuota BTC_y. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 6. classification::dummy_most_frequent

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `dummy_most_frequent` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
DummyClassifier determinista que siempre pronostica la clase más frecuente de train. No tiene red, capas, activaciones ni regularización.

2. OPTIMIZACIÓN
No optimiza parámetros: sin loss entrenada, optimizador, learning rate, batch o épocas.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.333352           0.333333  0.166674 24.028443    18215   18215
5_extreme_4_5 holdout_train  0.303321           0.200000  0.093092 25.110841    18215   18215
  5_quintiles holdout_train  0.200000           0.200000  0.066667 28.834923    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.381404           0.333333  0.184066 22.296471    18215    7808
5_extreme_4_5 holdout  0.341957           0.200000  0.101928 23.718275    18215    7808
  5_quintiles holdout  0.232838           0.200000  0.075545 27.651317    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy     macro_f1           log_loss         
                  mean      std              mean std     mean      std      mean      std
    3_regimes 0.333392 0.000050          0.333333 0.0 0.166688 0.000019 24.027005 0.001789
5_extreme_4_5 0.303359 0.000076          0.200000 0.0 0.093101 0.000018 25.109489 0.002746
  5_quintiles 0.200060 0.000043          0.200000 0.0 0.066683 0.000012 28.832755 0.001539

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy     macro_f1           log_loss         
                  mean      std              mean std     mean      std      mean      std
    3_regimes 0.289621 0.127296          0.333333 0.0 0.145638 0.051750 25.604651 4.588210
5_extreme_4_5 0.294234 0.091703          0.200000 0.0 0.089710 0.021652 25.438387 3.305295
  5_quintiles 0.179242 0.087444          0.200000 0.0 0.059319 0.024967 29.583111 3.151808

Métricas por clase en holdout:
regime_scheme   class_name  precision  recall       f1  support
    3_regimes          LOW   0.381404     1.0 0.552197     2978
    3_regimes          MID   0.000000     0.0 0.000000     2643
    3_regimes         HIGH   0.000000     0.0 0.000000     2187
5_extreme_4_5  EXTREME_LOW   0.000000     0.0 0.000000      435
5_extreme_4_5          LOW   0.341957     1.0 0.509639     2670
5_extreme_4_5          MID   0.000000     0.0 0.000000     2399
5_extreme_4_5         HIGH   0.000000     0.0 0.000000     2074
5_extreme_4_5 EXTREME_HIGH   0.000000     0.0 0.000000      230
  5_quintiles     VERY_LOW   0.232838     1.0 0.377727     1818
  5_quintiles          LOW   0.000000     0.0 0.000000     1690
  5_quintiles          MID   0.000000     0.0 0.000000     1604
  5_quintiles         HIGH   0.000000     0.0 0.000000     1390
  5_quintiles    VERY_HIGH   0.000000     0.0 0.000000     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH                0  2187    0
LOW                 0  2978    0
MID                 0  2643    0

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                0            0     0   230    0
EXTREME_LOW                 0            0     0   435    0
HIGH                        0            0     0  2074    0
LOW                         0            0     0  2670    0
MID                         0            0     0  2399    0

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH                0    0    0          0      1390
LOW                 0    0    0          0      1690
MID                 0    0    0          0      1604
VERY_HIGH           0    0    0          0      1306
VERY_LOW            0    0    0          0      1818

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: la entrada se ignora; el benchmark usa únicamente la frecuencia de clases de train. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 7. classification::tree_volatility

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `tree_volatility` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
DecisionTreeClassifier CART, max_depth=4 y min_samples_leaf=250. La profundidad y el mínimo por hoja actúan como regularización estructural; no hay capas neuronales, activaciones, Dropout ni BatchNorm.

2. OPTIMIZACIÓN
Criterio Gini de scikit-learn; crecimiento voraz del árbol; random_state=42. No usa optimizador por gradiente, learning rate, batch size ni épocas.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.465056           0.465047  0.456849  1.024043    18215   18215
5_extreme_4_5 holdout_train  0.406094           0.267765  0.253530  1.279020    18215   18215
  5_quintiles holdout_train  0.305298           0.305298  0.263114  1.522516    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.459016           0.452771  0.444597  1.024964    18215    7808
5_extreme_4_5 holdout  0.404841           0.262473  0.249435  1.286036    18215    7808
  5_quintiles holdout  0.306352           0.303603  0.262675  1.531010    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.489079 0.021202          0.489059 0.021198 0.469017 0.023792 0.996936 0.026343
5_extreme_4_5 0.428515 0.018416          0.282570 0.012158 0.246123 0.024133 1.250588 0.028575
  5_quintiles 0.328336 0.017562          0.328278 0.017544 0.281247 0.034145 1.491489 0.029779

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.464514 0.079922          0.431968 0.028045 0.411834 0.036571 1.029186 0.081917
5_extreme_4_5 0.389720 0.024277          0.248537 0.010565 0.212883 0.018642 1.309627 0.052425
  5_quintiles 0.312290 0.090016          0.274056 0.024599 0.234609 0.017141 1.525618 0.103298

Métricas por clase en holdout:
regime_scheme   class_name  precision   recall       f1  support
    3_regimes          LOW   0.494558 0.625588 0.552409     2978
    3_regimes          MID   0.356883 0.259932 0.300788     2643
    3_regimes         HIGH   0.488658 0.472794 0.480595     2187
5_extreme_4_5  EXTREME_LOW   0.000000 0.000000 0.000000      435
5_extreme_4_5          LOW   0.435057 0.567041 0.492358     2670
5_extreme_4_5          MID   0.332295 0.311380 0.321498     2399
5_extreme_4_5         HIGH   0.432692 0.433944 0.433317     2074
5_extreme_4_5 EXTREME_HIGH   0.000000 0.000000 0.000000      230
  5_quintiles     VERY_LOW   0.336579 0.532453 0.412441     1818
  5_quintiles          LOW   0.240585 0.321302 0.275146     1690
  5_quintiles          MID   0.000000 0.000000 0.000000     1604
  5_quintiles         HIGH   0.230849 0.160432 0.189304     1390
  5_quintiles    VERY_HIGH   0.385020 0.503828 0.436484     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH             1034   600  553
LOW               430  1863  685
MID               652  1304  687

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                0            0   150    24   56
EXTREME_LOW                 0            0    41   287  107
HIGH                        0            0   900   583  591
LOW                         0            0   409  1514  747
MID                         0            0   580  1072  747

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH              223  413    0        368       386
LOW               186  543    0        205       756
MID               218  510    0        315       561
VERY_HIGH         170  273    0        658       205
VERY_LOW          169  518    0        163       968

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: sigma_past. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 8. classification::tree_topology

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `tree_topology` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
DecisionTreeClassifier CART, max_depth=4 y min_samples_leaf=250. La profundidad y el mínimo por hoja actúan como regularización estructural; no hay capas neuronales, activaciones, Dropout ni BatchNorm.

2. OPTIMIZACIÓN
Criterio Gini de scikit-learn; crecimiento voraz del árbol; random_state=42. No usa optimizador por gradiente, learning rate, batch size ni épocas.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.420917           0.420904  0.402023  1.069309    18215   18215
5_extreme_4_5 holdout_train  0.371672           0.245068  0.216445  1.333342    18215   18215
  5_quintiles holdout_train  0.265770           0.265770  0.205769  1.575810    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.430584           0.418443  0.399075  1.055513    18215    7808
5_extreme_4_5 holdout  0.387935           0.250294  0.224851  1.309397    18215    7808
  5_quintiles holdout  0.266393           0.256534  0.191267  1.564759    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.435495 0.021064          0.435468 0.021041 0.407122 0.023860 1.056610 0.016860
5_extreme_4_5 0.387529 0.021136          0.255541 0.013971 0.222053 0.020147 1.319431 0.017878
  5_quintiles 0.280081 0.013940          0.280042 0.013924 0.242196 0.028936 1.562565 0.016607

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.401186 0.041591          0.390560 0.029028 0.352004 0.026692 1.088184 0.029373
5_extreme_4_5 0.358023 0.034252          0.230135 0.014251 0.193693 0.013338 1.372716 0.126913
  5_quintiles 0.234926 0.024492          0.243676 0.017192 0.192546 0.016629 1.601340 0.030688

Métricas por clase en holdout:
regime_scheme   class_name  precision   recall       f1  support
    3_regimes          LOW   0.453970 0.673942 0.542506     2978
    3_regimes          MID   0.359317 0.183125 0.242607     2643
    3_regimes         HIGH   0.426961 0.398262 0.412113     2187
5_extreme_4_5  EXTREME_LOW   0.000000 0.000000 0.000000      435
5_extreme_4_5          LOW   0.406985 0.641573 0.498038     2670
5_extreme_4_5          MID   0.342105 0.157149 0.215367     2399
5_extreme_4_5         HIGH   0.376051 0.452748 0.410851     2074
5_extreme_4_5 EXTREME_HIGH   0.000000 0.000000 0.000000      230
  5_quintiles     VERY_LOW   0.274921 0.720572 0.397995     1818
  5_quintiles          LOW   0.260870 0.014201 0.026936     1690
  5_quintiles          MID   0.200276 0.090399 0.124570     1604
  5_quintiles         HIGH   0.173134 0.041727 0.067246     1390
  5_quintiles    VERY_HIGH   0.286998 0.415773 0.339587     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH              871   896  420
LOW               528  2007  443
MID               641  1518  484

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                0            0   128    78   24
EXTREME_LOW                 0            0    83   302   50
HIGH                        0            0   939   824  311
LOW                         0            0   617  1713  340
MID                         0            0   730  1292  377

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH               58   12  130        393       797
LOW                74   24  155        301      1136
MID                60   24  145        379       996
VERY_HIGH          88   13  136        543       526
VERY_LOW           55   19  158        276      1310

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: BTC_y y dBTC_y. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 9. classification::tree_volatility_plus_topology

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `tree_volatility_plus_topology` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
DecisionTreeClassifier CART, max_depth=4 y min_samples_leaf=250. La profundidad y el mínimo por hoja actúan como regularización estructural; no hay capas neuronales, activaciones, Dropout ni BatchNorm.

2. OPTIMIZACIÓN
Criterio Gini de scikit-learn; crecimiento voraz del árbol; random_state=42. No usa optimizador por gradiente, learning rate, batch size ni épocas.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.470107           0.470099  0.465128  1.018168    18215   18215
5_extreme_4_5 holdout_train  0.411419           0.271276  0.257750  1.273967    18215   18215
  5_quintiles holdout_train  0.305682           0.305682  0.264833  1.517347    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.471055           0.465973  0.461873  1.020539    18215    7808
5_extreme_4_5 holdout  0.416752           0.271648  0.258539  1.280405    18215    7808
  5_quintiles holdout  0.308786           0.305264  0.262726  1.528947    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.496610 0.023806          0.496590 0.023799 0.486490 0.029722 0.986824 0.027967
5_extreme_4_5 0.436804 0.022692          0.288035 0.014982 0.270718 0.012068 1.240850 0.029722
  5_quintiles 0.333519 0.021043          0.333476 0.021021 0.304133 0.035945 1.480040 0.032615

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.461812 0.070711          0.433951 0.036615 0.420735 0.038987 1.031218 0.078576
5_extreme_4_5 0.393476 0.020342          0.250597 0.013112 0.232242 0.006973 1.314410 0.060001
  5_quintiles 0.310906 0.096573          0.274192 0.026265 0.245905 0.018331 1.529941 0.098270

Métricas por clase en holdout:
regime_scheme   class_name  precision   recall       f1  support
    3_regimes          LOW   0.506446 0.606783 0.552093     2978
    3_regimes          MID   0.380775 0.308740 0.340995     2643
    3_regimes         HIGH   0.503100 0.482396 0.492530     2187
5_extreme_4_5  EXTREME_LOW   0.000000 0.000000 0.000000      435
5_extreme_4_5          LOW   0.450583 0.549813 0.495277     2670
5_extreme_4_5          MID   0.345758 0.336390 0.341010     2399
5_extreme_4_5         HIGH   0.441787 0.472035 0.456410     2074
5_extreme_4_5 EXTREME_HIGH   0.000000 0.000000 0.000000      230
  5_quintiles     VERY_LOW   0.327612 0.608911 0.426015     1818
  5_quintiles          LOW   0.258238 0.227219 0.241737     1690
  5_quintiles          MID   0.000000 0.000000 0.000000     1604
  5_quintiles         HIGH   0.231753 0.221583 0.226554     1390
  5_quintiles    VERY_HIGH   0.379417 0.468606 0.419322     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH             1055   548  584
LOW               428  1807  743
MID               614  1213  816

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                0            0   157    21   52
EXTREME_LOW                 0            0    39   279  117
HIGH                        0            0   979   496  599
LOW                         0            0   443  1468  759
MID                         0            0   598   994  807

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH              308  254    0        345       483
LOW               235  384    0        196       875
MID               294  330    0        300       680
VERY_HIGH         262  198    0        612       234
VERY_LOW          230  321    0        160      1107

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: sigma_past, BTC_y y dBTC_y. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 10. classification::lightgbm_volatility_multiscale

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `lightgbm_volatility_multiscale` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
LGBMClassifier multiclase: gradient boosting de 300 árboles, num_leaves=15 y max_depth=5. La restricción de hojas/profundidad regulariza la complejidad; no es una red neuronal y no usa Dropout o BatchNorm.

2. OPTIMIZACIÓN
Objective=multiclass con log-loss; learning_rate=0.03; n_estimators=300; random_state=42; sin L1/L2 explícitos. No existe batch size; cada árbol es una iteración.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.589020           0.589015  0.587919  0.865180    18215   18215
5_extreme_4_5 holdout_train  0.553006           0.420298  0.448268  1.029043    18215   18215
  5_quintiles holdout_train  0.493659           0.493659  0.491033  1.268017    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.505379           0.505225  0.501723  0.970973    18215    7808
5_extreme_4_5 holdout  0.443648           0.294750  0.282028  1.223863    18215    7808
  5_quintiles holdout  0.330815           0.332840  0.329595  1.473400    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.690961 0.089455          0.690954 0.089453 0.690423 0.089689 0.732129 0.107323
5_extreme_4_5 0.680508 0.121418          0.600369 0.168402 0.641588 0.164407 0.832006 0.161678
  5_quintiles 0.649219 0.132412          0.649207 0.132404 0.648634 0.132725 1.049040 0.182390

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.511499 0.077558          0.463881 0.028243 0.467906 0.031114 0.972222 0.098920
5_extreme_4_5 0.434992 0.012167          0.269301 0.008241 0.259602 0.008815 1.270562 0.061645
  5_quintiles 0.343328 0.075591          0.298595 0.021204 0.300432 0.022465 1.482799 0.099434

Métricas por clase en holdout:
regime_scheme   class_name  precision   recall       f1  support
    3_regimes          LOW   0.561965 0.595366 0.578184     2978
    3_regimes          MID   0.392812 0.351495 0.371006     2643
    3_regimes         HIGH   0.543706 0.568816 0.555978     2187
5_extreme_4_5  EXTREME_LOW   0.000000 0.000000 0.000000      435
5_extreme_4_5          LOW   0.479815 0.583146 0.526458     2670
5_extreme_4_5          MID   0.351094 0.307628 0.327927     2399
5_extreme_4_5         HIGH   0.477636 0.561234 0.516072     2074
5_extreme_4_5 EXTREME_HIGH   0.227273 0.021739 0.039683      230
  5_quintiles     VERY_LOW   0.396665 0.431793 0.413484     1818
  5_quintiles          LOW   0.285254 0.262130 0.273204     1690
  5_quintiles          MID   0.242063 0.228180 0.234917     1604
  5_quintiles         HIGH   0.241176 0.235971 0.238545     1390
  5_quintiles    VERY_HIGH   0.470798 0.506126 0.487823     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH             1244   330  613
LOW               382  1773  823
MID               662  1052  929

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                5            0   170    23   32
EXTREME_LOW                 2            0    45   314   74
HIGH                       10            0  1164   353  547
LOW                         3            2   397  1557  711
MID                         2            0   661   998  738

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH              328  224  340        305       193
LOW               222  443  350        119       556
MID               324  335  366        210       369
VERY_HIGH         298  105  166        661        76
VERY_LOW          188  446  290        109       785

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: estado multiescala causal de volatilidad: nivel, lags, medias, dispersión y EMA. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 11. classification::lightgbm_topology_multiscale

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `lightgbm_topology_multiscale` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
LGBMClassifier multiclase: gradient boosting de 300 árboles, num_leaves=15 y max_depth=5. La restricción de hojas/profundidad regulariza la complejidad; no es una red neuronal y no usa Dropout o BatchNorm.

2. OPTIMIZACIÓN
Objective=multiclass con log-loss; learning_rate=0.03; n_estimators=300; random_state=42; sin L1/L2 explícitos. No existe batch size; cada árbol es una iteración.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.581444           0.581438  0.578659  0.943660    18215   18215
5_extreme_4_5 holdout_train  0.559319           0.411583  0.433211  1.103696    18215   18215
  5_quintiles holdout_train  0.534779           0.534779  0.534001  1.338723    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.460169           0.444324  0.428693  1.052158    18215    7808
5_extreme_4_5 holdout  0.397925           0.254627  0.234238  1.317945    18215    7808
  5_quintiles holdout  0.276639           0.268929  0.251976  1.571786    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.708300 0.109868          0.708296 0.109871 0.707726 0.110167 0.805885 0.136804
5_extreme_4_5 0.723693 0.129552          0.643409 0.193353 0.680993 0.182125 0.884642 0.196327
  5_quintiles 0.709849 0.142800          0.709840 0.142800 0.710364 0.142544 1.108438 0.225798

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.410148 0.038571          0.397906 0.018539 0.379130 0.013210 1.106330 0.051084
5_extreme_4_5 0.352026 0.041589          0.231951 0.011378 0.214303 0.012359 1.437597 0.219651
  5_quintiles 0.250280 0.025565          0.242659 0.016755 0.227759 0.005918 1.638708 0.058806

Métricas por clase en holdout:
regime_scheme   class_name  precision   recall       f1  support
    3_regimes          LOW   0.473938 0.726662 0.573701     2978
    3_regimes          MID   0.393540 0.225880 0.287019     2643
    3_regimes         HIGH   0.482319 0.380430 0.425358     2187
5_extreme_4_5  EXTREME_LOW   0.105263 0.004598 0.008811      435
5_extreme_4_5          LOW   0.410223 0.703371 0.518212     2670
5_extreme_4_5          MID   0.340176 0.193414 0.246612     2399
5_extreme_4_5         HIGH   0.413232 0.367406 0.388974     2074
5_extreme_4_5 EXTREME_HIGH   0.333333 0.004348 0.008584      230
  5_quintiles     VERY_LOW   0.290960 0.575358 0.386477     1818
  5_quintiles          LOW   0.234811 0.169231 0.196699     1690
  5_quintiles          MID   0.205882 0.104738 0.138843     1604
  5_quintiles         HIGH   0.217871 0.156115 0.181894     1390
  5_quintiles    VERY_HIGH   0.374472 0.339204 0.355966     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH              832   878  477
LOW               371  2164  443
MID               522  1524  597

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                1            0   123    59   47
EXTREME_LOW                 0            2    39   335   59
HIGH                        1            1   762   886  424
LOW                         1           13   408  1878  370
MID                         0            3   512  1420  464

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH              217  195  173        249       556
LOW               177  286  160        152       915
MID               214  285  168        196       741
VERY_HIGH         240  133  153        443       337
VERY_LOW          148  319  162        143      1046

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: estado multiescala causal de BTC_y: nivel, cambios, Shannon, logit, lags, rolling y EMA. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```

## 12. classification::lightgbm_volatility_plus_topology

```text
Actuá como revisor metodológico de modelos predictivos para series temporales financieras.

Analizá el modelo `lightgbm_volatility_plus_topology` sin inventar componentes y usando exclusivamente la ficha y métricas siguientes.

1. ARQUITECTURA
LGBMClassifier multiclase: gradient boosting de 300 árboles, num_leaves=15 y max_depth=5. La restricción de hojas/profundidad regulariza la complejidad; no es una red neuronal y no usa Dropout o BatchNorm.

2. OPTIMIZACIÓN
Objective=multiclass con log-loss; learning_rate=0.03; n_estimators=300; random_state=42; sin L1/L2 explícitos. No existe batch size; cada árbol es una iteración.

3. MÉTRICAS
Entrenamiento del ajuste final:
regime_scheme         split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout_train  0.627121           0.627116  0.626375  0.836412    18215   18215
5_extreme_4_5 holdout_train  0.598133           0.478301  0.520874  0.974564    18215   18215
  5_quintiles holdout_train  0.577820           0.577820  0.576221  1.207688    18215   18215

Holdout cronológico:
regime_scheme   split  accuracy  balanced_accuracy  macro_f1  log_loss  n_train  n_eval
    3_regimes holdout  0.512295           0.511870  0.505532  0.967205    18215    7808
5_extreme_4_5 holdout  0.443776           0.294090  0.278778  1.227082    18215    7808
  5_quintiles holdout  0.337218           0.339281  0.331197  1.472161    18215    7808

Train walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.749158 0.102404          0.749153 0.102403 0.748857 0.102458 0.677085 0.128560
5_extreme_4_5 0.755659 0.127509          0.712034 0.180297 0.748420 0.158394 0.738160 0.186516
  5_quintiles 0.748420 0.141409          0.748413 0.141407 0.747931 0.141803 0.942144 0.222138

Validación walk-forward, media y desvío:
regime_scheme accuracy          balanced_accuracy          macro_f1          log_loss         
                  mean      std              mean      std     mean      std     mean      std
    3_regimes 0.515058 0.076356          0.469870 0.029011 0.471209 0.029264 0.972759 0.100806
5_extreme_4_5 0.435453 0.018248          0.272449 0.007197 0.261920 0.010441 1.285950 0.090462
  5_quintiles 0.335354 0.074780          0.294916 0.020531 0.294622 0.022882 1.486640 0.102313

Métricas por clase en holdout:
regime_scheme   class_name  precision   recall       f1  support
    3_regimes          LOW   0.568259 0.619208 0.592640     2978
    3_regimes          MID   0.401277 0.332955 0.363937     2643
    3_regimes         HIGH   0.538397 0.583448 0.560018     2187
5_extreme_4_5  EXTREME_LOW   0.166667 0.002299 0.004535      435
5_extreme_4_5          LOW   0.474209 0.605993 0.532062     2670
5_extreme_4_5          MID   0.359892 0.276782 0.312912     2399
5_extreme_4_5         HIGH   0.465613 0.567985 0.511729     2074
5_extreme_4_5 EXTREME_HIGH   0.266667 0.017391 0.032653      230
  5_quintiles     VERY_LOW   0.409932 0.463146 0.434917     1818
  5_quintiles          LOW   0.280866 0.284024 0.282436     1690
  5_quintiles          MID   0.235450 0.166459 0.195033     1604
  5_quintiles         HIGH   0.256612 0.258273 0.257440     1390
  5_quintiles    VERY_HIGH   0.453042 0.524502 0.486160     1306

Matrices de confusión holdout:
3_regimes:
predicted_class  HIGH   LOW  MID
true_class                      
HIGH             1276   338  573
LOW               394  1844  740
MID               700  1063  880

5_extreme_4_5:
predicted_class  EXTREME_HIGH  EXTREME_LOW  HIGH   LOW  MID
true_class                                                 
EXTREME_HIGH                4            0   178    19   29
EXTREME_LOW                 1            1    43   332   58
HIGH                        7            0  1178   406  483
LOW                         3            4   434  1618  611
MID                         0            1   697  1037  664

5_quintiles:
predicted_class  HIGH  LOW  MID  VERY_HIGH  VERY_LOW
true_class                                          
HIGH              359  249  251        340       191
LOW               240  480  263        131       576
MID               318  408  267        245       366
VERY_HIGH         283  111  148        685        79
VERY_LOW          199  461  205        111       842

4. CONTEXTO DE LOS DATOS
Problema: clasificación temporal de 3 regímenes, 5 regímenes con colas 4.5% y 5 quintiles. Features: familias multiescala de volatilidad y BTC_y más interacciones entre ambas. Preprocesamiento: alineación cronológica, transformaciones causales, eliminación explícita de NaN/inf, holdout final del 30%, purga h=2, sin shuffle y umbrales ajustados sólo en train. Los árboles no requieren escalado ni codificación one-hot.

Tarea: redactá una ficha técnica rigurosa; compará train contra validación y test; diagnosticá
posible sobreajuste o subajuste; explicá el equilibrio entre clases; distinguí desempeño
predictivo de causalidad; y cerrá con fortalezas, límites y siguiente prueba recomendada.
Si un campo típico de redes neuronales no aplica, indicá explícitamente por qué.
```
