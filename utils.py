from torch.utils.data import Dataset
from PIL import Image

class MyDataset1(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        """
        txt _ path : txt text path that contains path information for the image, along with label information
        transform : data processing, random cropping of images, and conversion to tensor
        Multi-task label
        """
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), float(words[2]), int(words[3])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label2, label1, label3 = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape, label)
        return img, label2, label1, label3

    def __len__(self):
        return len(self.imgs)


class MyDataset2(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        """
        txt _ path : txt text path that contains path information for the image, along with label information
        transform : data processing, random cropping of images, and conversion to tensor
        Single-task label
        """
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[3])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape, label)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetCluster(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        """
        Variante de MyDataset2 para select_clusters.py: usa la CLASE real
        de supervivencia (words[1]: 0=corto, 1=largo) como etiqueta,
        en vez del índice de paciente (words[3]) que usa MyDataset2.
        """
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
