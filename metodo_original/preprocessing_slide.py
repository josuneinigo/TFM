import openslide
import numpy as np
from openslide.deepzoom import DeepZoomGenerator
import imageio
import os

BASE   = '/home/jinigo/HSNP_code'
WSI_DIR = '/data/projects/jinigo/WSIs2'

width  = 1024
highth = 1024

for root, dirs, files in os.walk(WSI_DIR):
    for fname in files:
        if not fname.endswith('.svs'):
            continue
        svs_path    = os.path.join(root, fname)
        patient_id  = '-'.join(fname.split('-')[:3])   # extrae TCGA-XX-XXXX
        result_path = os.path.join(BASE, 'data', 'parches', patient_id)
        os.makedirs(result_path, exist_ok=True)

        print(f"Procesando: {fname}")
        slide = openslide.open_slide(svs_path)
        [w, h] = slide.level_dimensions[0]
        print(f"  Dimensiones: {w} x {h}")

        data_gen = DeepZoomGenerator(slide, tile_size=highth, overlap=0, limit_bounds=False)

        num_w = int(np.floor(w / width)) + 1
        num_h = int(np.floor(h / highth)) + 1
        for i in range(num_w):
            vID = '{:0>2}'.format(i + 1)
            for j in range(num_h):
                pID = '{:0>2}'.format(j + 1)
                fi  = patient_id + '_' + vID + pID + '.png'   # prefijo con ID paciente
                f   = os.path.join(result_path, fi)
                img = np.array(data_gen.get_tile(data_gen.level_count - 1, (i, j)))
                summ = np.count_nonzero(img)
                img1 = img.copy()
                img1[img1 > 230] = 255
                num  = np.count_nonzero(img1 == 255)
                mm   = summ - num
                if mm > 500000:
                    imageio.imsave(f, img)

        slide.close()
        print(f"  Completado: {patient_id}")
