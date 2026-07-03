import glob, os
import cv2

BASE       = '/home/jinigo/HSNP_code'
INPUT_DIR  = os.path.join(BASE, 'data', 'parches')
OUTPUT_DIR = os.path.join(BASE, 'data', 'parches_gray')

def timage():
    for files in glob.glob(os.path.join(INPUT_DIR, '**', '*.png'), recursive=True):
        filepath, filename = os.path.split(files)
        filtername, exts   = os.path.splitext(filename)

        rel_path   = os.path.relpath(filepath, INPUT_DIR)
        out_subdir = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(out_subdir, exist_ok=True)

        im         = cv2.imread(files)
        gray       = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        GrayImage  = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(out_subdir, f"{filtername}.png"), GrayImage)

if __name__ == '__main__':
    timage()
    print('Conversion completed')
