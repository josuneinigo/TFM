"""
OS_time_prediction_test_phikon.py
Evaluacion final del modelo Phikon sobre el conjunto de TEST del cluster 05.
Carga dual_inceptionv3_phikon_best.pth y calcula RMSE, MAE y CC sobre
los parches de test del cluster 05 de Phikon-v2.

Uso:
    CUDA_VISIBLE_DEVICES=0 python OS_time_prediction_test_phikon.py
"""
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model.dual_inceptionv3 import Dual_InceptionV3

BASE = '/home/jinigo/HSNP_code'
TEST_PATH = os.path.join(BASE, 'data', 'labels', 'K8_phikon', '05_test.txt')
CKPT_PATH = os.path.join(BASE, 'checkpoints', 'dual_inceptionv3',
                         'dual_inceptionv3_phikon_best.pth')
OUT_PATH  = os.path.join(BASE, 'reporting', 'dual_inceptionv3',
                         'result_phikon_test.txt')


class MyDatasetTest(Dataset):
    def __init__(self, path, transform=None):
        self.imgs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                self.imgs.append((words[0], float(words[2]), int(words[3])))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fn, os_time, pidx = self.imgs[idx]
        img = Image.open(fn).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os_time, pidx


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f'Cargando checkpoint: {CKPT_PATH}')
    net = Dual_InceptionV3()
    net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    net.to(device)
    net.eval()

    dataset = MyDatasetTest(TEST_PATH, transform=transform)
    print(f'Parches de test: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, os_times, _ in loader:
            images = images.to(device)
            output1, output2, aux1, aux2 = net(images)
            output1 = output1.to(torch.float64).cpu().numpy()
            all_preds.extend(output1.flatten())
            all_labels.extend(os_times.numpy().flatten())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    mae  = np.mean(np.abs(all_preds - all_labels))
    cc   = np.corrcoef(all_preds, all_labels)[0, 1] \
           if np.std(all_preds) > 0 and np.std(all_labels) > 0 else float('nan')

    print()
    print('=== Resultado final Phikon-v2, cluster 05, TEST ===')
    print(f'  N.º parches: {len(all_labels)}')
    print(f'  RMSE: {rmse:.3f} días')
    print(f'  MAE:  {mae:.3f} días')
    print(f'  CC:   {cc:.3f}')

    with open(OUT_PATH, 'w') as f:
        f.write(f'RMSE: {rmse:.3f}\n')
        f.write(f'MAE:  {mae:.3f}\n')
        f.write(f'CC:   {cc:.3f}\n')
        f.write(f'N parches test: {len(all_labels)}\n')
    print(f'Guardado en: {OUT_PATH}')


if __name__ == '__main__':
    main()
