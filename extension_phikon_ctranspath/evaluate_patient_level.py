"""
evaluate_patient_level.py

Evalua un checkpoint ya entrenado de select_clusters.py agregando las
predicciones POR PACIENTE (voto mayoritario de sus parches), en vez de
contar parches sueltos. Esto reduce el ruido que aparece cuando un
unico paciente con muchos parches domina el accuracy global.

No reentrena nada: solo carga el modelo guardado y hace inferencia
sobre el conjunto de validacion correspondiente a ese cluster.

Uso:
    python evaluate_patient_level.py 01
    python evaluate_patient_level.py 00 02 03 04 05 06 07   (varios a la vez)
"""

import os
import sys
from collections import defaultdict, Counter

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model.resnet import resnet50

BASE = '/home/jinigo/HSNP_code'


class MyDatasetClusterEval(Dataset):
    """Como MyDatasetCluster, pero tambien devuelve el indice de paciente
    (words[3]) para poder agregar las predicciones por paciente."""
    def __init__(self, txt_path, transform=None):
        self.imgs = []
        with open(txt_path, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                if not line:
                    continue
                words = line.split()
                # ruta, clase(0/1), os_days, pidx
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

    txt_path = os.path.join(BASE, 'data', 'labels', 'K=8', f'{cluster}_val.txt')
    ckpt_path = os.path.join(BASE, 'checkpoints', 'resnet_classification', f'{cluster}_resnet50.pth')

    if not os.path.exists(ckpt_path):
        print(f"[{cluster}] Checkpoint no encontrado todavia: {ckpt_path}")
        return None

    dataset = MyDatasetClusterEval(txt_path, transform=val_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    net = resnet50(num_classes=2)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device)
    net.eval()

    # Acumula, por paciente: lista de predicciones de parche + la etiqueta real
    patient_preds = defaultdict(list)
    patient_true = {}
    correct_patches = 0
    total_patches = 0

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
                correct_patches += int(pred == label)
                total_patches += 1

    # Agregacion por paciente: voto mayoritario de sus parches
    correct_patients = 0
    total_patients = len(patient_preds)
    detalle = []
    for pidx, preds in patient_preds.items():
        voto_mayoritario = Counter(preds).most_common(1)[0][0]
        acierto = int(voto_mayoritario == patient_true[pidx])
        correct_patients += acierto
        detalle.append((pidx, voto_mayoritario, patient_true[pidx], len(preds), acierto))

    acc_parche = correct_patches / total_patches if total_patches else 0.0
    acc_paciente = correct_patients / total_patients if total_patients else 0.0

    print(f"\n=== Cluster {cluster} ===")
    print(f"  Accuracy a nivel de PARCHE   : {acc_parche:.3f}  ({correct_patches}/{total_patches})")
    print(f"  Accuracy a nivel de PACIENTE : {acc_paciente:.3f}  ({correct_patients}/{total_patients})")
    print(f"  Detalle por paciente (pidx, pred_mayoritaria, real, n_parches, acierto):")
    for fila in sorted(detalle):
        print(f"    {fila}")

    return {"cluster": cluster, "acc_parche": acc_parche, "acc_paciente": acc_paciente,
            "n_pacientes": total_patients}


def main():
    if len(sys.argv) < 2:
        print("Uso: python evaluate_patient_level.py <cluster> [<cluster> ...]")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    resultados = []
    for cluster in sys.argv[1:]:
        r = evaluate_cluster(cluster, device)
        if r:
            resultados.append(r)

    if resultados:
        print("\n=== Resumen ===")
        for r in resultados:
            print(f"  Cluster {r['cluster']}: acc_parche={r['acc_parche']:.3f}  "
                  f"acc_paciente={r['acc_paciente']:.3f}  (n={r['n_pacientes']} pacientes)")


if __name__ == '__main__':
    main()
