"""
evaluate_patient_level_valtest.py

Extiende evaluate_patient_level.py combinando los pacientes de
VALIDACION y de TEST para tener una muestra mayor (y por tanto menos
ruidosa) al calcular el accuracy por paciente de cada cluster.

No reentrena nada: usa los checkpoints ya guardados por
select_clusters.py. Los ficheros de test/ (uno por paciente, con
parches de los 8 clusters mezclados) se filtran aqui mismo por
cluster, leyendo el cluster directamente de la ruta de cada imagen
(data/image_all/<cluster>/...).

Uso:
    python evaluate_patient_level_valtest.py 00 01 02 03 04 05 06 07
"""

import os
import sys
import glob
from collections import defaultdict, Counter

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model.resnet import resnet50

BASE = '/home/jinigo/HSNP_code'


def cluster_from_path(path):
    """Extrae el id de cluster ('00'..'07') de una ruta tipo
    .../data/image_all/04/TCGA-xx.png"""
    parts = path.replace('\\', '/').split('/')
    for i, p in enumerate(parts):
        if p == 'image_all' and i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_lines(txt_path):
    lines = []
    with open(txt_path, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            if line:
                lines.append(line)
    return lines


def build_combined_lines(cluster):
    """Junta las lineas de <cluster>_val.txt con las lineas de
    test/*.txt que pertenezcan a ese cluster (filtrando por ruta)."""
    combined = []

    val_path = os.path.join(BASE, 'data', 'labels', 'K=8', f'{cluster}_val.txt')
    if os.path.exists(val_path):
        combined.extend(load_lines(val_path))

    test_files = glob.glob(os.path.join(BASE, 'data', 'labels', 'test', '*.txt'))
    for tf in test_files:
        for line in load_lines(tf):
            ruta = line.split()[0]
            if cluster_from_path(ruta) == cluster:
                combined.append(line)

    return combined


class MyDatasetFromLines(Dataset):
    def __init__(self, lines, transform=None):
        self.imgs = []
        for line in lines:
            words = line.split()
            self.imgs.append((words[0], int(words[1]), int(words[3])))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fn, label, pidx = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, pidx


def evaluate_cluster(cluster, device):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ckpt_path = os.path.join(BASE, 'checkpoints', 'resnet_classification', f'{cluster}_resnet50.pth')
    if not os.path.exists(ckpt_path):
        print(f"[{cluster}] Checkpoint no encontrado: {ckpt_path}")
        return None

    lines = build_combined_lines(cluster)
    if not lines:
        print(f"[{cluster}] Sin datos val+test para este cluster")
        return None

    dataset = MyDatasetFromLines(lines, transform=val_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    net = resnet50(num_classes=2)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device)
    net.eval()

    patient_preds = defaultdict(list)
    patient_true = {}

    with torch.no_grad():
        for images, labels, pidxs in loader:
            images = images.to(device)
            outputs = net(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels = labels.tolist()
            pidxs = pidxs.tolist()
            for pred, label, pidx in zip(preds, labels, pidxs):
                patient_preds[pidx].append(pred)
                patient_true[pidx] = label

    correct_patients = 0
    total_patients = len(patient_preds)
    detalle = []
    for pidx, preds in patient_preds.items():
        voto = Counter(preds).most_common(1)[0][0]
        acierto = int(voto == patient_true[pidx])
        correct_patients += acierto
        detalle.append((pidx, voto, patient_true[pidx], len(preds), acierto))

    acc_paciente = correct_patients / total_patients if total_patients else 0.0

    print(f"\n=== Cluster {cluster} (val+test combinados) ===")
    print(f"  Accuracy a nivel de PACIENTE : {acc_paciente:.3f}  ({correct_patients}/{total_patients})")
    for fila in sorted(detalle):
        print(f"    {fila}")

    return {"cluster": cluster, "acc_paciente": acc_paciente, "n_pacientes": total_patients}


def main():
    if len(sys.argv) < 2:
        print("Uso: python evaluate_patient_level_valtest.py <cluster> [<cluster> ...]")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    resultados = []
    for cluster in sys.argv[1:]:
        r = evaluate_cluster(cluster, device)
        if r:
            resultados.append(r)

    if resultados:
        print("\n=== Resumen (val+test) ===")
        for r in resultados:
            sel = "Si" if r['acc_paciente'] >= 0.55 else "No"
            print(f"  Cluster {r['cluster']}: acc_paciente={r['acc_paciente']:.3f}  "
                  f"(n={r['n_pacientes']} pacientes)  Seleccionado(b=0.55): {sel}")


if __name__ == '__main__':
    main()
