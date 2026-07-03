from PIL import Image
import glob, os

BASE       = '/home/jinigo/HSNP_code'
INPUT_DIR  = os.path.join(BASE, 'data', 'parches_gray')
OUTPUT_DIR = os.path.join(BASE, 'data', 'parches_resize')

def timage():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for files in glob.glob(os.path.join(INPUT_DIR, '**', '*.png'), recursive=True):
        filepath, filename  = os.path.split(files)
        filtername, exts    = os.path.splitext(filename)
        im                  = Image.open(files)
        im_ss               = im.resize((28, 28))
        im_ss.save(os.path.join(OUTPUT_DIR, filtername + '.png'))

if __name__ == '__main__':
    timage()
    print('Conversion completed')
