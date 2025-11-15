# IA_P3_CIFAR10_PerezRegueiroMiguel

# CIFAR-10 CNN – Práctica 3  
> Autor: TuNombre Apellido – Curso 2025/26

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)

Repositorio reproducible de la práctica **“Visión profunda con CNN en CIFAR-10”**.  
Incluye: notebooks, curvas, matrices de confusión y estudio de ablación.

## 📦 Estructura
```
IA_P3_CIFAR10_Apellido/
├── notebooks/
│   └── CIFAR10_CNN_Apellido.ipynb   # notebook principal (colab)
├── results/
│   ├── data_meta.json               # hash y formas de datos
│   ├── params.yaml                  # hiper-parámetros
│   ├── history_*.csv                # curvas de entrenamiento
│   └── metrics_*.json               # accuracies finales
├── figuras/
│   ├── muestras_cifar10.png
│   ├── confusion_matrix_cnn3.png
│   ├── errores_cnn3.png
│   └── curvas_resumen.png
├── env/
│   ├── ENVIRONMENT.md               # versiones
│   └── requirements.txt             # pip freeze
└── README.md                        # este archivo
```

# A) Conceptos clave – Visión profunda con CNN en CIFAR-10

## Mapa rápido del tema
CIFAR-10 son 60 000 imágenes pequeñas (32×32 píxeles y 3 canales de color) divididas en 10 clases: avión, coche, pájaro, gato, venado, perro, rana, caballo, barco y camión.  
Una CNN supera a una red densa porque **no aplana** la imagen: mantiene la estructura 2D y usa **convoluciones** para detectar bordes, texturas y formas. El **pooling** añade **invarianza a pequeñas traslaciones**: si un gato se mueve unos píxeles, el filtro sigue activándose. Aplanar los 3 072 píxeles al principio obligaría a la red a aprender de memoria la posición exacta de cada píxel, con **800 000 parámetros** solo en la primera capa, y **sensible al ruido de fondo**.

## Convolución sin magia
Un filtro (kernel) es una matriz pequeña (ej. 3×3) que se desliza sobre la imagen.  
Hiper-parámetros: **tamaño**, **stride** (paso), **padding** (borde) y **canales** (profundidad).  
Ejemplo rápido: imagen 5×5×1, kernel 3×3, stride=1, padding=0 → salida 3×3×1.  
Coste: 9×3×3 = 81 multiplicaciones por canal; si usamos 32 filtros → 2 592 ops.

## Pooling y por qué importa
**MaxPooling** conserva el valor máximo dentro de una ventana (2×2): preserva bordes fuertes y reduce ruido.  
**AveragePooling** suaviza, útil en fondos homogéneos.  
Ambos **dividen a la mitad** la resolución, aumentan la **invarianza traslacional** y **disminuyen sobreajuste** al reducir parámetros.  
Micro-ejemplo: ventana 2×2 con valores [[4,2],[3,6]] → Max=6, Average=3.75.

## Arquitectura típica de una CNN simple
Input(32×32×3)  
→ Conv2D(32 filtros, 3×3) + ReLU (detecta bordes)  
→ MaxPool2D(2×2) (reduce a 16×16)  
→ Conv2D(64 filtros, 3×3) + ReLU (formas complejas)  
→ MaxPool2D(2×2) (8×8)  
→ Flatten (aplanado solo al final)  
→ Dense(128) + ReLU (combinación global)  
→ Dropout(0.5) (regularización)  
→ Dense(10, softmax) (probabilidades por clase)

## Métrica y pérdida adecuadas
**Pérdida**: `categorical_crossentropy` (etiquetas one-hot).  
**Métrica principal**: `accuracy` (% aciertos).  
**Matriz de confusión**: muestra qué clases se confunden (ej. *cat ↔ dog*), útil para detectar sesgos o clases difíciles.



## Normalización y preparación de datos
Dividimos por 255.0 para llevar píxeles a [0,1] → gradientes estables y LR más altas.  
**Estandarizar por canal** (media 0, desv 1) acelera convergencia en redes profundas o con SGD+momentum.  
Ambas mejoran la **estabilidad numérica** y permiten usar **tasas de aprendizaje más grandes** sin divergencia.

## Baseline denso vs CNN
MLP: 3072→256→10 → ≈800 k parámetros, **sin sesgo espacial**, **sobreajusta** rápido ante ruido de fondo.  
CNN: 55 k parámetros, **sesgo inductivo local** (vecinos → patrones), **generaliza** mejor con menos datos y parámetros.  
La CNN **no aplan** la imagen → conserva topología y es **más robusta** a pequeñas deformaciones.

## Parámetros y capacidad
Conv2D:  
`parámetros = (kernel_h × kernel_w × canales_entrada + 1) × filtros_salida`  
Aumentar **kernel**, **filtros** o **profundidad** → más capacidad, más tiempo y riesgo de sobreajuste.  
Profundidad crece capacidad **exponencialmente**; conviene equilibrar con regularización.

## Regularización práctica
1. **Dropout**: apaga neuronas (0.2-0.5) → evita co-adaptación.  
2. **L2 weight decay**: penaliza pesos grandes (1e-4) → pesos más pequeños.  
3. **Data Augmentation**: crea variedad artificial → robustez.  
4. **Early Stopping**: para cuando val_loss no mejora → ahorra tiempo y evita sobreajuste.  
**Combina** las 3 primeras; EarlyStopping siempre obligado.

## Data Augmentation con cabeza
Plan razonable CIFAR-10:  
- Flip horizontal (siempre).  
- Rotación ±10°.  
- Traslación 10 %.  
- Zoom 10 %.  
- Brillo ±20 %.  
**Límites**: CIFAR-10 ya es natural → evita distorsiones extremas, rotaciones &gt;20° o cambios de color fuertes.

## Optimización y LR scheduling
**Adam**: adaptativo, rápido, pero puede quedarse en mínimos locales.  
**SGD+momentum**: más lento, a veces **mejor generalización**.  
**ReduceLROnPlateau**: baja LR cuando val_loss se estanca 3 épocas.  
**CosineDecay**: baja LR suavemente de 0.05 → 0 en 30 épocas.  
**Señal**: val_loss sin mejora → bajar LR.

## Curvas de aprendizaje
- **Subajuste**: train/val altas y paralelas → aumenta capacidad.  
- **Ajuste saludable**: brecha pequeña y descendente.  
- **Sobreajuste**: train baja, val sube → más regularización.

## Matriz de confusión y clase difícil
Pares típicos: *cat ↔ dog*, *automobile ↔ truck*, *deer ↔ horse*.  
Mejoras: más datos de esas clases (augment dirigido), **label smoothing** o **focal loss**.

## Batch size y estabilidad
- **32**: ruido útil, generaliza mejor, época lenta.  
- **128**: estable, época rápida, pero puede necesitar más épocas.  
**Valor inicial en Colab**: 64 (equilibrio tiempo/ruido).

## Buenas prácticas de entrega
1. Código limpio y comentado.  
2. Semillas fijadas (42).  
3. Logs completos (history.csv, metrics.json).  
4. Curvas y matriz de confusión.  
5. Tabla comparativa MLP vs CNN.  
6. README con instrucciones de reproducción.  
7. `requirements.txt` congelado.  
8. Tag de release (`v1.0-P3-CIFAR10_Apellido`).  
9. Informe PDF (2 págs).  
10. 5 hallazgos breves (ej. “augment +2.3 % test acc”).



## 🏃‍♂️ Uso rápido
1. Clona y crea entorno:
```bash
git clone https://github.com/tu-usuario/IA_P3_CIFAR10_Apellido.git
cd IA_P3_CIFAR10_Apellido
python -m venv venv && source venv/bin/activate
pip install -r env/requirements.txt
```
2. Abre el notebook en Colab/Jupyter y ejecuta **Run all**.

## 📊 Resultados clave (resumen)
| Modelo                     | test acc | épocas | parámetros | notas |
|----------------------------|----------|--------|------------|-------|
| MLP (baseline)             | 0.XX     | 10     | 800 k      | overfit fuerte |
| CNN-2B                     | 0.XX     | 15     | 55 k       | — |
| CNN-2B + L2 + EarlyStop    | 0.XX     | XX     | 55 k       | brecha ↓ |
| CNN-3B + augment + sched   | **0.XX** | XX     | 200 k      | **mejor** |
| SGD + CosineDecay          | 0.XX     | XX     | 200 k      | similar, estable |

> Mejora final sobre MLP: **+XX %** con **4× menos parámetros**.

## 🔍 Ablación (contribución de cada técnica)
| Variante        | test acc | Δ vs todo |
|-----------------|----------|-----------|
| A todo          | 0.XX     | —         |
| B sin augment   | 0.XX     | -0.XX     |
| C sin L2        | 0.XX     | -0.XX     |
| D sin dropout   | 0.XX     | -0.XX     |

**Conclusión**: *Data augmentation* es la técnica **más influyente**.

## 🧪 Reproducibilidad
| elemento        | valor                        |
|-----------------|------------------------------|
| seed            | 42                           |
| TensorFlow      | 2.15.0 (GPU habilitado)      |
| Python          | 3.10.12                      |
| commit          | `abc1234`                    |
| tag             | v1.0-P3-CIFAR10_Apellido     |
| hash datos      | `b5a2c1d8e7f9a1b2` (SHA-256) |

## ✍️ Próximos pasos
- Transfer learning con ResNet-20 → objetivo 92 %.  
- Label smoothing / MixUp para reducir confusión *cat ↔ dog*.  
- Auto-augment para ganar generalización extra.

## 📄 Licencia
Este trabajo académico se distribuye bolely under **CC BY-NC-SA 4.0**.
```

Cómo usarlo:

1. En GitHub → Add file → Create new file → pega el texto → nombre `README.md` → Commit.  
2. Cambia los `0.XX` por los valores que ya tienes en los `metrics_*.json`.  
3. Añade el tag y ya tienes un **repo profesional** listo para entregar.
