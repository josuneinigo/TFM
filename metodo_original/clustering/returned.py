import os
import shutil

BASE      = '/home/jinigo/HSNP_code'
PARCHES   = os.path.join(BASE, 'data', 'parches')    # imágenes originales (por subcarpeta de paciente)
CLUSTERS  = os.path.join(BASE, 'data', 'clusters')   # resultado del K-Means (00..07)
IMAGE_ALL = os.path.join(BASE, 'data', 'image_all')  # destino: imágenes originales reorganizadas por cluster

# Construye índice: nombre_parche -> ruta_original
index = {}
for patient in os.listdir(PARCHES):
    patient_path = os.path.join(PARCHES, patient)
    if not os.path.isdir(patient_path):
        continue
    for fname in os.listdir(patient_path):
        index[fname] = os.path.join(patient_path, fname)

print(f"Index: {len(index)} parches indexados")

# Copia cada imagen original a su carpeta de cluster correspondiente
total_copied = 0
for cluster_name in os.listdir(CLUSTERS):
    cluster_src = os.path.join(CLUSTERS, cluster_name)
    if not os.path.isdir(cluster_src):
        continue
    cluster_dst = os.path.join(IMAGE_ALL, cluster_name)
    os.makedirs(cluster_dst, exist_ok=True)
    cluster_files = os.listdir(cluster_src)
    print(f"Cluster {cluster_name}: {len(cluster_files)} archivos en clusters/")
    copied = 0
    missing = 0
    for fname in cluster_files:
        if fname in index:
            shutil.move(index[fname], os.path.join(cluster_dst, fname))
            copied += 1
        else:
            missing += 1
    print(f"  -> copiados: {copied}  no encontrados en index: {missing}")
    total_copied += copied

print(f"\nTotal copiado: {total_copied}")
