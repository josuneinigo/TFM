import os
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from model.dual_inceptionv3 import Dual_InceptionV3
from utils import MyDataset1

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    print("using {} device.".format(device))
    data_transform = {
        "val": transforms.Compose([transforms.Resize(299),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    BASE = r'C:\Users\iosun\Desktop\MASTER\LOGICA, COMPUTACION E INTELIGENCIA ARTIFICIAL\TFM\CODIGO'
    txt_path = os.path.join(BASE, 'data', 'labels', 'test')
    filenames = os.listdir(txt_path)
    filenames.sort()
    f = open(os.path.join(BASE, 'reporting', 'dual_inceptionv3', '100-dual_inceptionv3.txt'), 'w')
    for filename in filenames:
        pic_paths1 = txt_path + '/' + filename
        print(pic_paths1)
        validate_dataset = MyDataset1(pic_paths1, transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=nw)

        print("using {} images for validation.".format(val_num))

        net = Dual_InceptionV3()
        net.to(device)

        weights_path = os.path.join(BASE, 'checkpoints', 'dual_inceptionv3', '100-dual_inceptionv3.pth')
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        net.load_state_dict(torch.load(weights_path))

        net.eval()
        mse1 = 0.0
        mae1 = 0.0
        a = 0.0
        x = 0.0
        acc = 0.0  # accumulate accurate number / epoch

        with torch.no_grad():

            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                valimages, vallabels2, vallabels1, vallabels3 = val_data
                valoutput1, valoutput2,aux1,aux2 = net(valimages.to(device))
                valoutput1 = valoutput1.to(torch.float64)


                vallabels1 = vallabels1.data.cpu().numpy()
                valoutput1 = valoutput1.data.cpu().numpy()
                x += np.sum(valoutput1)
                mse1 += np.sum((valoutput1 - vallabels1) ** 2)
                mae1 += np.sum(np.abs(valoutput1 - vallabels1))
                predict2_y = torch.max(valoutput2, dim=1)[1]
                acc += torch.eq(predict2_y, vallabels2.to(device)).sum().item()
                predict2_y1 = F.softmax(valoutput2, dim=1)
                a += torch.sum(predict2_y1, dim=0)

        MSE = mse1 / val_num
        RMSE = MSE ** 0.5
        MAE = mae1 / val_num
        val2_accurate = acc / val_num
        asum = a / val_num
        asum =torch.unsqueeze(asum ,0)

        asum1 = torch.max(asum, dim=1)[1]
        x = x / val_num

        print('test1_rmse: %.3f  test2_accurate: %.3f' %
              (RMSE, val2_accurate))
        print('asum:', asum)
        print('asum1:', asum1)
        print('x:', x)
        print('MAE:', MAE)
        out_path = filename + ' ' + str(round(float(MSE), 4)) + ' ' + str(round(float(RMSE), 4))+ ' ' + str(round(float(MAE), 4)) + ' ' + str(round(float(val2_accurate), 4))+' '+str(round(float(x), 4))+ ' ' + str(asum)+' ' + str(asum1)
        f.write(out_path)
        f.write("\n ")
    f.close()


if __name__ == '__main__':
    main()