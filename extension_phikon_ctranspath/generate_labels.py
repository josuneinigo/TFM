"""
Genera los ficheros .txt de etiquetas necesarios para select_clusters.py
y OS_time_prediction_train.py a partir del clinical.tsv del TCGA.

Ejecutar DESPUÉS de haber completado las 4 fases de clustering.

Formato de cada línea en los .txt:
  ruta_imagen  clase  dias_supervivencia  indice_paciente
  - clase: 0=supervivencia corta (<365 días), 1=larga (>=365 días)
"""

import os
import csv
import random

BASE        = '/home/jinigo/HSNP_code'
CLINICAL    = os.path.join(BASE, 'data', 'labels', 'clinical.tsv')
IMAGE_ALL   = os.path.join(BASE, 'data', 'image_all')   # parches organizados por cluster
LABELS_DIR  = os.path.join(BASE, 'data', 'labels')

THRESHOLD   = 365   # días: < umbral = corto (0), >= umbral = largo (1)
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── 1. Leer datos clínicos ──────────────────────────────────────────────────

clinical = {}   # patient_id -> (os_days, clase, idx)

with open(CLINICAL, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    idx = 0
    for row in reader:
        pid         = row['cases.submitter_id'].strip()
        vital       = row['demographic.vital_status'].strip()
        days_death  = row['demographic.days_to_death'].strip().strip("'")
        days_follow = row['diagnoses.days_to_last_follow_up'].strip().strip("'")

        # OS time: días hasta muerte si fallecido, último seguimiento si vivo
        if vital == 'Dead' and days_death not in ('--', ''):
            os_days = float(days_death)
        elif days_follow not in ('--', ''):
            os_days = float(days_follow)
        else:
            continue   # sin datos de supervivencia, omitir

        clase = 0 if os_days < THRESHOLD else 1
        clinical[pid] = (os_days, clase, idx)
        idx += 1

print(f"Pacientes con datos clínicos válidos: {len(clinical)}")

# ── 2. Reunir todos los parches con su etiqueta ────────────────────────────

all_patches = []   # lista de (ruta, clase, os_days, indice_paciente, cluster_id)

for cluster_name in sorted(os.listdir(IMAGE_ALL)):
    cluster_path = os.path.join(IMAGE_ALL, cluster_name)
    if not os.path.isdir(cluster_path):
        continue
    for fname in os.listdir(cluster_path):
        if not fname.endswith('.png'):
            continue
        pid = fname.split('_')[0]   # extrae TCGA-XX-XXXX del prefijo
        if pid not in clinical:
            continue
        os_days, clase, pidx = clinical[pid]
        ruta = os.path.join(cluster_path, fname)
        all_patches.append((ruta, clase, os_days, pidx, cluster_name))

print(f"Total parches con etiqueta: {len(all_patches)}")

# ── 3. Split train / val / test por paciente ───────────────────────────────

pids_con_datos = list({p[3] for p in all_patches})
random.seed(42)
random.shuffle(pids_con_datos)

n       = len(pids_con_datos)
n_val   = int(n * VAL_RATIO)
n_test  = int(n * TEST_RATIO)

val_pids  = set(pids_con_datos[:n_val])
test_pids = set(pids_con_datos[n_val:n_val + n_test])
train_pids = set(pids_con_datos[n_val + n_test:])

print(f"Pacientes — train: {len(train_pids)}  val: {len(val_pids)}  test: {len(test_pids)}")

# ── 4. Generar ficheros para select_clusters.py (K=8) ─────────────────────

k8_dir = os.path.join(LABELS_DIR, 'K=8')
os.makedirs(k8_dir, exist_ok=True)

# Un fichero por cluster (solo datos de train)
cluster_files = {}
for ruta, clase, os_days, pidx, cluster_name in all_patches:
    if pidx not in train_pids:
        continue
    if cluster_name not in cluster_files:
        cluster_files[cluster_name] = []
    cluster_files[cluster_name].append(f"{ruta} {clase} {os_days} {pidx}")

for cluster_name, lines in cluster_files.items():
    out = os.path.join(k8_dir, f"{cluster_name}.txt")
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Escrito: {out}  ({len(lines)} parches)")

# Ficheros de validación POR CLUSTER para select_clusters.py
# (misma lógica que train: cada cluster se valida solo con su propia distribución)
val_cluster_files = {}
for ruta, clase, os_days, pidx, cluster_name in all_patches:
    if pidx not in val_pids:
        continue
    val_cluster_files.setdefault(cluster_name, []).append(f"{ruta} {clase} {os_days} {pidx}")

for cluster_name, lines in val_cluster_files.items():
    out = os.path.join(k8_dir, f"{cluster_name}_val.txt")
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Escrito: {out}  ({len(lines)} parches)")

# ── 5. Generar train.txt / val.txt para OS_time_prediction_train.py ────────
#    (usa todos los clusters — después de select_clusters.py filtrarás manualmente
#     los clusters no relevantes o regeneras con solo los relevantes)

train_lines = [
    f"{ruta} {clase} {os_days} {pidx}"
    for ruta, clase, os_days, pidx, _ in all_patches
    if pidx in train_pids
]
val_lines_os = [
    f"{ruta} {clase} {os_days} {pidx}"
    for ruta, clase, os_days, pidx, _ in all_patches
    if pidx in val_pids
]

with open(os.path.join(LABELS_DIR, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_lines))
with open(os.path.join(LABELS_DIR, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_lines_os))

# Carpeta test/ con un .txt por paciente
test_dir = os.path.join(LABELS_DIR, 'test')
os.makedirs(test_dir, exist_ok=True)
test_by_pid = {}
for ruta, clase, os_days, pidx, _ in all_patches:
    if pidx in test_pids:
        test_by_pid.setdefault(pidx, []).append(f"{ruta} {clase} {os_days} {pidx}")
for pidx, lines in test_by_pid.items():
    with open(os.path.join(test_dir, f"{pidx}.txt"), 'w') as f:
        f.write('\n'.join(lines))

print(f"\nEtiquetas generadas en: {LABELS_DIR}")
print(f"  train.txt: {len(train_lines)} parches")
print(f"  val.txt:   {len(val_lines_os)} parches")
print(f"  test/:     {len(test_by_pid)} pacientes")
