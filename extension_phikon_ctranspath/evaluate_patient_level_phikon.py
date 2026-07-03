"""
evaluate_patient_level_phikon.py
Evalua los checkpoints de select_clusters.py entrenados sobre los clusters
de Phikon-v2 (K8_phikon), agregando las predicciones por paciente
(voto mayoritario) sobre la union de val+test de cada cluster.
Identico en logica a evaluate_patient_level_ctranspath.py, solo cambian
las rutas de labels y de checkpoints.
Uso:
    python evaluate_patient_level_phikon.py 01 02 04 05 06 07
(El cluster 00 no tiene val/test, y el 03 no genero checkpoint por el mismo
motivo -- no incluirlos en la lista de argumentos.)
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
LABELS_DIR = os.path.join(BASE, 'data', 'labels', 'K8_phikon')
CKPT_DIR = os.path.join(BASE, 'checkpoints', 'resnet_classification')


def load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.rstrip() for l in f if l.rstrip()]


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
    ckpt_path = os.path.join(CKPT_DIR, f'K8_phikon_{cluster}_resnet50.pth')
    if not os.path.exists(ckpt_path):
        print(f"[{cluster}] Checkpoint no encontrado: {ckpt_path}")
        return None
    lines = load_lines(os.path.join(LABELS_DIR, f'{cluster}_val.txt'))
    lines += load_lines(os.path.join(LABELS_DIR, f'{cluster}_test.txt'))
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
    for pidx, preds in patient_preds.items():
        voto = Counter(preds).most_common(1)[0][0]
        correct_patients += int(voto == patient_true[pidx])
    acc_paciente = correct_patients / total_patients if total_patients else 0.0
    print(f"  Cluster {cluster}: acc_paciente={acc_paciente:.3f}  "
          f"({correct_patients}/{total_patients} pacientes, {len(lines)} parches val+test)")
    return {"cluster": cluster, "acc_paciente": acc_paciente, "n_pacientes": total_patients}


def main():
    if len(sys.argv) < 2:
        print("Uso: python evaluate_patient_level_phikon.py <cluster> [<cluster> ...]")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}\n")
    resultados = []
    for cluster in sys.argv[1:]:
        r = evaluate_cluster(cluster, device)
        if r:
            resultados.append(r)
    if resultados:
        print("\n=== Resumen Phikon-v2 (val+test) ===")
        for r in resultados:
            sel = "Si" if r['acc_paciente'] >= 0.55 else "No"
            print(f"  Cluster {r['cluster']}: acc_paciente={r['acc_paciente']:.3f}  "
                  f"(n={r['n_pacientes']} pacientes)  Seleccionado(b=0.55): {sel}")


if __name__ == '__main__':
    main()
