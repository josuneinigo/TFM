r"""
extract_features_ctranspath.py

Extrae features de 768-D con CTransPath (Wang et al., Medical Image
Analysis 2022) a partir de parches de histopatología en formato PNG.

Carga los pesos oficiales (rehospedados en HuggingFace Hub por la
comunidad, idénticos a los del repo Xiyue-Wang/TransPath) usando una
versión moderna de `timm`, sin necesidad de clonar el repo original
ni de fijar timm==0.5.4.

Requisitos (entorno conda 'hsnp'):
    pip install timm huggingface_hub pillow numpy tqdm torchvision

Uso (PowerShell / CMD, Windows):
    python extract_features_ctranspath.py --input_dir "C:\datos\parches" --output_dir "C:\datos\salida"
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from timm.layers.helpers import to_2tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ----------------------------------------------------------------------
# 1. "ConvStem": el patch-embedding híbrido CNN + Swin que usa CTransPath
#    en lugar del patch-embedding estándar de un Swin-Tiny de timm.
#    Adaptado de Xiyue-Wang/TransPath/ctran.py (líneas 6-44).
# ----------------------------------------------------------------------
class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768,
                 norm_layer=None, **kwargs):
        super().__init__()
        assert patch_size == 4, "CTransPath exige patch_size = 4"
        assert embed_dim % 8 == 0, "embed_dim debe ser múltiplo de 8"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem += [
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ]
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Tamaño de entrada ({H}x{W}) no coincide con el esperado "
            f"({self.img_size[0]}x{self.img_size[1]})"
        )
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC (formato esperado por Swin en timm)
        x = self.norm(x)
        return x


# ----------------------------------------------------------------------
# 2. Dataset mínimo para leer PNG de un directorio (búsqueda recursiva)
# ----------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.paths = sorted(Path(root_dir).rglob("*.png"))
        if not self.paths:
            raise RuntimeError(f"No se encontraron .png en {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, str(path)


def build_model(device):
    """Carga CTransPath en modo extracción de features (sin cabeza de clasificación)."""
    model = timm.create_model(
        model_name="hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
        embed_layer=ConvStem,
        pretrained=True,
        num_classes=0,  # sin head -> forward() devuelve el embedding de 768-D
    )
    model.eval().to(device)
    return model


def build_transform():
    # Mismas estadísticas de normalización (ImageNet) y tamaño (224x224)
    # usados en el preentrenamiento de CTransPath.
    #
    # IMPORTANTE: si tus parches de 1024x1024 se extrajeron a una
    # magnificación/mpp distinta a la usada en el preentrenamiento (~20x),
    # un Resize directo cambia el campo de visión efectivo. Ajusta el
    # recorte/downsampling antes de llegar aquí si es necesario.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


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

    model = build_model(device)
    transform = build_transform()
    dataset = PatchDataset(args.input_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                         num_workers=args.num_workers, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)

    all_feats, all_names = [], []
    for imgs, paths in tqdm(loader, desc="CTransPath"):
        imgs = imgs.to(device)
        feats = model(imgs)  # (B, 768)
        all_feats.append(feats.cpu().numpy())
        all_names.extend(paths)

    all_feats = np.concatenate(all_feats, axis=0)
    np.save(os.path.join(args.output_dir, "ctranspath_features.npy"), all_feats)
    with open(os.path.join(args.output_dir, "ctranspath_patch_names.txt"), "w") as f:
        f.write("\n".join(all_names))

    print(f"Listo. {all_feats.shape[0]} parches -> features de dimensión {all_feats.shape[1]}")
    print(f"Guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
