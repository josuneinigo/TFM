from PIL import Image
from pylab import *
from scipy.cluster.vq import *
import os
import glob
from sklearn.decomposition import PCA

BASE       = '/home/jinigo/HSNP_code' 
INPUT_DIR  = os.path.join(BASE, 'data', 'parches_resize')
OUTPUT_DIR = os.path.join(BASE, 'data', 'clusters')

imlist = glob.glob(os.path.join(INPUT_DIR, '*.png'))
imnbr  = len(imlist)

# Load images, run PCA.
X = array([array(Image.open(im)).flatten() for im in imlist], 'f')

p = PCA(n_components=50)
a = p.fit_transform(X)
print(a.shape)

k = 8
# create feature vector from k first eigenvectors #
# by stacking eigenvectors as columns
features = array(a[:,:k])
print('features:',features.shape)

# k-means
centroids, distortion = kmeans(features,k)
code, distance = vq(features, centroids)

# save each image to its cluster folder
for c in range(k):
    ind = where(code == c)[0]
    pre_savename = os.path.join(OUTPUT_DIR, '%02d' % c)
    os.makedirs(pre_savename, exist_ok=True)
    for i in range(minimum(len(ind), 40000)):
        im = Image.open(imlist[ind[i]])
        image_name = os.path.basename(imlist[ind[i]])
        image_name_real = os.path.splitext(image_name)[0]
        image_name_new = image_name_real + ".png"
        savename = os.path.join(pre_savename, image_name_new)
        im.save(savename)
    print(f"  Cluster {c:02d}: {len(ind)} imagenes")
