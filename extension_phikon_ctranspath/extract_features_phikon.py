r"""
extract_features_phikon.py
Extrae features de 1024-D con Phikon-v2 (Filiot et al., 2024,
arXiv:2409.09173) a partir de parches de histopatología en formato PNG.

A diferencia de CTransPath (timm + ConvStem custom), Phikon-v2 es un
ViT-Large estándar cargado vía la librería `transformers` (AutoModel +
AutoImageProcessor), sin necesidad de ningún módulo de patch-embedding
personalizado.

Modelo: ViT-Large (0.3B params), preentrenado con DINOv2 sobre PANCAN-XL
(~450M imágenes de histopatología a 20x). Salida: vector de 1024-D
(CLS token de la última capa oculta).

Licencia: Owkin non-commercial licence (ver
https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf) — uso
académico/no comercial, adecuado para este TFM.

Requisitos (entorno conda 'hsnp'):
    pip install transformers pillow numpy tqdm torch --break-system-packages
    (NO requiere timm ni torchvision para este script, a diferencia del
    de CTransPath, ya que el preprocesamiento lo gestiona AutoImageProcessor)

Uso (mismo patrón de CLI que extract_features_ctranspath.py):
    CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/hsnp/bin/python extract_features_phikon.py \
        --input_dir data/image_all --output_dir data/features_phikon --batch_size 32
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "owkin/phikon-v2"


# ----------------------------------------------------------------------
# Dataset mínimo para leer PNG de un directorio (búsqueda recursiva).
# Idéntico al de extract_features_ctranspath.py: devuelve la imagen PIL
# SIN transformar todavía, porque aquí el preprocesamiento lo aplica
# el AutoImageProcessor de Phikon-v2 (resize/normalize propios), no una
# torchvision.transforms.Compose manual.
# ----------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = sorted(Path(root_dir).rglob("*.png"))
        if not self.paths:
            raise RuntimeError(f"No se encontraron .png en {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return img, str(path)


def collate_keep_pil(batch):
    """Collate manual: NO se puede usar el collate por defecto de PyTorch
    porque las imágenes PIL no son tensores. Devolvemos listas y dejamos
    que el AutoImageProcessor haga el batching en el bucle principal."""
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)


def build_model(device):
    """Carga Phikon-v2 en modo extracción de features (CLS token)."""
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval().to(device)
    return model


def build_processor():
    return AutoImageProcessor.from_pretrained(MODEL_ID)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Carpeta con parches PNG")
    parser.add_argument("--output_dir", required=True, help="Carpeta de salida")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Cargando {MODEL_ID} (ViT-Large, ~1.2GB)...")
    model = build_model(device)
    processor = build_processor()

    dataset = PatchDataset(args.input_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_keep_pil,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_feats, all_names = [], []
    for imgs, paths in tqdm(loader, desc="Phikon-v2"):
        inputs = processor(imgs, return_tensors="pt").to(device)
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0, :]  # CLS token -> (B, 1024)
        all_feats.append(feats.cpu().numpy())
        all_names.extend(paths)

    all_feats = np.concatenate(all_feats, axis=0)
    np.save(os.path.join(args.output_dir, "phikon_features.npy"), all_feats)
    with open(os.path.join(args.output_dir, "phikon_patch_names.txt"), "w") as f:
        f.write("\n".join(all_names))

    print(f"Listo. {all_feats.shape[0]} parches -> features de dimensión {all_feats.shape[1]}")
    print(f"Guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
