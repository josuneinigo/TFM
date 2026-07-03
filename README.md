# Predicción del tiempo de supervivencia global (OS) en glioblastoma

Código del Trabajo Fin de Máster (TFM) del Máster en Lógica, Computación e
Inteligencia Artificial. El proyecto parte del pipeline de **Dai et al.
(2022)** — descrito más abajo — y lo amplía sustituyendo su etapa de
clustering de parches histológicos por dos *foundation models* de
histopatología (**Phikon-v2** y **CTransPath**), además de introducir varias
mejoras en el entrenamiento y la evaluación.

> **Importante sobre la autoría.** Este repositorio combina dos partes
> claramente separadas en carpetas distintas:
>
> - [`metodo_original/`](metodo_original/): código de los autores del
>   artículo original (ver cita más abajo), movido de sitio para poder
>   organizar el repositorio pero **sin modificar su lógica**. Todo el
>   mérito de esta parte es suyo.
> - [`extension_phikon_ctranspath/`](extension_phikon_ctranspath/): código
>   propio desarrollado para este TFM, que amplía el trabajo anterior.
>
> Los directorios `model/` (arquitecturas `ResNet50` y `Dual_InceptionV3`) y
> `utils.py` (clases `Dataset`) son compartidos por ambas partes.

## Cita del trabajo original

El pipeline de partida (preprocesado de WSIs, clustering de parches por
fenotipo, selección de clusters relevantes para la supervivencia y
predicción final del tiempo de OS con una red `Dual_InceptionV3`) proviene
de:

> Dai, J., Qi, F., Gong, G., Liu, X., Li, D., & Xue, J. (2022).
> *Hypergraph-based spiking neural P systems for predicting the overall
> survival time of glioblastoma patients.* Expert Systems with
> Applications. https://doi.org/10.1016/j.eswa.2022.119234

El README original de los autores (instrucciones de uso de su pipeline) se
conserva íntegro en [`metodo_original/README.md`](metodo_original/README.md).

## Qué se ha ampliado en este TFM

El artículo original agrupa los parches de cada paciente en 8 clusters
mediante *k-means* sobre una representación en escala de grises de baja
dimensión de cada parche. En este TFM se sustituye (y compara) esa
representación por los *embeddings* de dos modelos preentrenados de
histopatología:

- **[Phikon-v2](https://huggingface.co/owkin/phikon-v2)** (Filiot et al.,
  2024, [arXiv:2409.09173](https://arxiv.org/abs/2409.09173)): ViT-Large
  preentrenado con DINOv2 sobre ~450M imágenes de histopatología.
  Licencia Owkin no comercial — uso académico.
- **[CTransPath](https://github.com/Xiyue-Wang/TransPath)** (Wang et al.,
  *Medical Image Analysis*, 2022): Swin-Transformer con *patch embedding*
  híbrido CNN, preentrenado con aprendizaje semi-supervisado contrastivo
  sobre WSIs.

Sobre esos embeddings se repite el mismo pipeline posterior (clustering,
selección de clusters por relevancia para la supervivencia, predicción de
OS con `Dual_InceptionV3`), lo que permite comparar el efecto de la
representación de partida sobre el resultado final. Además se añaden:

- Generación de etiquetas de supervivencia a partir de los metadatos
  clínicos de TCGA (`generate_labels.py`).
- Guardado del mejor checkpoint por *accuracy* de validación, no solo el de
  la última época (`select_clusters_best.py`).
- Una variante con regularización L2 del modelo de predicción de OS
  (`OS_time_prediction_train_regularizado.py`).
- Evaluación a nivel de paciente (voto mayoritario de sus parches) en vez
  de a nivel de parche suelto (`evaluate_patient_level*.py`).
- Generación de las gráficas de RMSE de validación a partir de los logs de
  entrenamiento (`generar_grafica_rmse.py`).

## Estructura del repositorio

```
.
├── metodo_original/            # Código de Dai et al. (2022), reubicado
│   ├── README.md                #   README original de los autores
│   ├── preprocessing_slide.py    #   Corte de WSIs (.svs) en parches
│   ├── clustering/                #   Clustering k-means por fenotipo
│   ├── select_clusters.py        #   Selección de clusters relacionados con OS
│   ├── OS_time_prediction_train.py
│   └── OS_time_prediction_test.py
├── extension_phikon_ctranspath/  # Ampliación propia del TFM
│   ├── generate_labels.py         #   Etiquetas OS a partir de TCGA clinical.tsv
│   ├── extract_features_phikon.py
│   ├── extract_features_ctranspath.py
│   ├── recluster_phikon.py       #   k-means sobre embeddings de Phikon-v2
│   ├── recluster_ctranspath.py   #   k-means sobre embeddings de CTransPath
│   ├── select_clusters_best.py
│   ├── evaluate_patient_level*.py
│   ├── OS_time_prediction_train_phikon.py
│   ├── OS_time_prediction_test_phikon.py
│   ├── OS_time_prediction_train_regularizado.py
│   └── generar_grafica_rmse.py
├── model/                        # Arquitecturas compartidas
│   ├── resnet.py                  #   ResNet50 (selección de clusters)
│   └── dual_inceptionv3.py        #   Dual_InceptionV3 (predicción de OS)
├── utils.py                      # Clases Dataset compartidas
├── reporting/                    # Resultados (accuracy, RMSE, gráficas)
└── checkpoints/                  # Pesos entrenados (no versionados, ver abajo)
```

## Requisitos

- Python 3.9, CUDA (se ha usado una GPU V100 16G)
- [OpenSlide](https://openslide.org/) (librería nativa + `openslide-python`)
- Paquetes Python: `torch`, `torchvision`, `numpy`, `scikit-learn`, `pillow`,
  `imageio`, `opencv-python`, `scipy`, `tqdm`, `matplotlib`
- Solo para la extensión: `transformers` (Phikon-v2) y `timm` +
  `huggingface_hub` (CTransPath)

## Datos

Se usan las cohortes GBM de The Cancer Genome Atlas (TCGA-GBM), incluyendo
las imágenes de histopatología (`.svs`) y los metadatos clínicos
(`clinical.tsv`), que **no se incluyen en este repositorio** por tamaño y
por ser datos de acceso público gestionado por TCGA. Los scripts asumen una
carpeta `data/` (no versionada) en la raíz del proyecto con la estructura
que generan los propios scripts del pipeline (`data/image_all`,
`data/labels`, `data/features_phikon`, `data/features_ctranspath`, etc.).

## Cómo ejecutar

Todos los scripts se lanzan como módulos desde la raíz del repositorio
(para que las importaciones a `model/` y `utils.py` funcionen).

### 1. Pipeline original (Dai et al., 2022)

```bash
python -m metodo_original.preprocessing_slide
python -m metodo_original.clustering.grayscale_processing
python -m metodo_original.clustering.picture_resize
python -m metodo_original.clustering.kmeans_clustering
python -m metodo_original.clustering.returned
python -m metodo_original.select_clusters            # train
python -m metodo_original.OS_time_prediction_train
python -m metodo_original.OS_time_prediction_test
```

### 2. Extensión propia (Phikon-v2 / CTransPath)

```bash
# Generar etiquetas de supervivencia a partir de TCGA clinical.tsv
python -m extension_phikon_ctranspath.generate_labels

# Extraer embeddings con el modelo de histopatología elegido
python -m extension_phikon_ctranspath.extract_features_phikon --input_dir data/image_all --output_dir data/features_phikon
python -m extension_phikon_ctranspath.extract_features_ctranspath --input_dir data/image_all --output_dir data/features_ctranspath

# Reclustering k-means sobre esos embeddings
python -m extension_phikon_ctranspath.recluster_phikon
python -m extension_phikon_ctranspath.recluster_ctranspath

# Selección de clusters relevantes (guardando el mejor checkpoint)
python -m extension_phikon_ctranspath.select_clusters_best <cluster> <labels_dir>

# Predicción final de OS (con o sin regularización L2)
python -m extension_phikon_ctranspath.OS_time_prediction_train_phikon
python -m extension_phikon_ctranspath.OS_time_prediction_test_phikon
python -m extension_phikon_ctranspath.OS_time_prediction_train_regularizado

# Evaluación a nivel de paciente
python -m extension_phikon_ctranspath.evaluate_patient_level_phikon <clusters...>
```

Antes de ejecutar, revisa la variable `BASE` al principio de cada script:
apunta a la ruta absoluta del proyecto en el cluster donde se entrenó
originalmente y hay que adaptarla a tu propia máquina.

## Checkpoints y resultados

Los pesos entrenados (`checkpoints/`, varios GB) y los logs de
entrenamiento no se incluyen en el repositorio (ver `.gitignore`). Los
resultados numéricos (accuracy por cluster, RMSE, etc.) sí están en
`reporting/`, y las gráficas generadas en `reporting/figures/`.

## Licencia

El código propio de este TFM se comparte con fines académicos. Ten en
cuenta que **Phikon-v2** se distribue bajo la licencia no comercial de
Owkin ([ver licencia](https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf)),
por lo que su uso queda restringido a fines de investigación/académicos.
