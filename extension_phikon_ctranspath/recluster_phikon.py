"""
recluster_phikon.py
A partir de los embeddings de Phikon-v2 (phikon_features.npy +
phikon_patch_names.txt), aplica k-means (k=8) y genera los ficheros de
labels por cluster (train/val/test), reutilizando el MISMO split de
pacientes que ya se uso para el baseline (train.txt, val.txt, test/*.txt),
para que la comparacion entre esquemas de preprocesamiento sea justa.
No mueve ni copia ninguna imagen: los ficheros de labels generados siguen
apuntando a la ubicacion fisica actual de cada parche (data/image_all/...).

Identico en logica a recluster_ctranspath.py, salvo las rutas de entrada
(features_phikon en vez de features_ctranspath) y de salida (K8_phikon en
vez de K8_ctranspath). La dimensionalidad del embedding (1024 en vez de
768) no requiere ningun cambio en el codigo: KMeans se adapta automaticamente
al numero de columnas de la matriz de entrada.

Uso:
    python recluster_phikon.py
"""
import os
import glob
import numpy as np
from sklearn.cluster import KMeans

BASE = '/home/jinigo/HSNP_code'
FEATURES_DIR = os.path.join(BASE, 'data', 'features_phikon')
OUT_DIR = os.path.join(BASE, 'data', 'labels', 'K8_phikon')
K = 8


def load_label_lines(path, split_name):
    lines = []
    if not os.path.exists(path):
        return lines
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line:
                words = line.split()
                lines.append((words[0], words[1], words[2], words[3], split_name))
    return lines


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Cargando embeddings...")
    feats = np.load(os.path.join(FEATURES_DIR, 'phikon_features.npy'))
    with open(os.path.join(FEATURES_DIR, 'phikon_patch_names.txt')) as f:
        # normalizamos a ruta absoluta: la extraccion pudo lanzarse con
        # --input_dir relativo, y los ficheros de labels usan rutas absolutas
        paths = [os.path.abspath(l.rstrip()) for l in f]
    print(f"  {feats.shape[0]} parches, {feats.shape[1]} dimensiones")

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feats_norm = feats / norms

    print("Ejecutando k-means (k=8)...")
    km = KMeans(n_clusters=K, random_state=42, n_init=3)
    cluster_ids = km.fit_predict(feats_norm)

    print("  Distribucion de clusters:")
    for c in range(K):
        print(f"    Cluster {c:02d}: {(cluster_ids == c).sum()} parches")

    path_to_cluster = {p: c for p, c in zip(paths, cluster_ids)}

    print("Cargando labels existentes (mismo split que el baseline)...")
    all_lines = []
    all_lines += load_label_lines(os.path.join(BASE, 'data', 'labels', 'train.txt'), 'train')
    all_lines += load_label_lines(os.path.join(BASE, 'data', 'labels', 'val.txt'), 'val')
    for tf in glob.glob(os.path.join(BASE, 'data', 'labels', 'test', '*.txt')):
        all_lines += load_label_lines(tf, 'test')
    print(f"  {len(all_lines)} lineas de labels en total")

    buckets = {(c, split): [] for c in range(K) for split in ('train', 'val', 'test')}
    no_match = 0
    for ruta, clase, os_days, pidx, split in all_lines:
        cluster = path_to_cluster.get(os.path.abspath(ruta))
        if cluster is None:
            no_match += 1
            continue
        buckets[(cluster, split)].append(f"{ruta} {clase} {os_days} {pidx}")
    print(f"  Parches sin cluster asignado (no encontrados en embeddings): {no_match}")

    for c in range(K):
        for split, suffix in (('train', ''), ('val', '_val'), ('test', '_test')):
            lines = buckets[(c, split)]
            out_path = os.path.join(OUT_DIR, f"{c:02d}{suffix}.txt")
            with open(out_path, 'w') as f:
                f.write('\n'.join(lines))
            print(f"  Escrito {out_path}: {len(lines)} parches")

    print("\nListo. Labels generados en:", OUT_DIR)


if __name__ == '__main__':
    main()
