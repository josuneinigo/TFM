import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from model.dual_inceptionv3 import Dual_InceptionV3
from utils import MyDataset1


def main():
    device = torch.device("cuda")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(299),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(299),
                                   transforms.CenterCrop(299),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    BASE = '/home/jinigo/HSNP_code'
    txt_path1 = os.path.join(BASE, 'data', 'labels', 'train_filtrado.txt')
    train_dataset = MyDataset1(txt_path1, transform=data_transform["train"])
    train_num = len(train_dataset)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    txt_path2 = os.path.join(BASE, 'data', 'labels', 'val_filtrado.txt')
    validate_dataset = MyDataset1(txt_path2, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = Dual_InceptionV3()
    net.to(device)

    loss1_function = nn.MSELoss()
    loss2_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    epochs = 100
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    with open(os.path.join(BASE, 'reporting', 'dual_inceptionv3', 'result.txt'), "w") as f:
        best_rmse = float('inf')
        lr_list = []
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            acc_t = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels2, labels1, labels3 = data
                optimizer.zero_grad()
                output1, output2,aux1,aux2 = net(images.to(device))
                output1 = output1.to(torch.float64)
                output2 = output2.to(torch.float64)
                aux1 = aux1.to(torch.float64)
                aux2 = aux2 .to(torch.float64)
                loss11 = loss1_function(output1, labels1.to(device))
                loss12 = loss1_function(aux1, labels1.to(device))
                loss1 = loss11+0.3*loss12
                loss1 = 0.00001 * loss1
                loss21 = loss2_function(output2, labels2.to(device))
                loss22 = loss2_function(aux2, labels2.to(device))
                loss2 = loss21+0.3*loss22
                loss = loss1 + loss2
                loss.backward()

                optimizer.step()
                train2_y = torch.max(output2, dim=1)[1]
                acc_t += torch.eq(train2_y, labels2.to(device)).sum().item()

                # print statistics
                running_loss += loss.item()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                train_acc = acc_t / train_num

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f} loss1:{:.3f} loss2:{:.3f}".format(epoch + 1,
                                                                                                   epochs,
                                                                                                   loss, loss1, loss2)

            # validate
            net.eval()
            mse1 = 0.0
            acc_v = 0.0
            valrunning_loss = 0.0
            valrunning_loss1 = 0.0
            valrunning_loss2 = 0.0
            with torch.no_grad():

                val_bar = tqdm(validate_loader, colour='green')
                for val_data in val_bar:
                    valimages, vallabels2, vallabels1, vallabels3 = val_data
                    valoutput1, valoutput2,aux1,aux2 = net(valimages.to(device))
                    valoutput1 = valoutput1.to(torch.float64)

                    valloss1 = loss1_function(valoutput1, vallabels1.to(device))
                    valloss1 = 0.00001 * valloss1
                    valloss2 = loss2_function(valoutput2, vallabels2.to(device))
                    valloss = valloss1 + valloss2

                    vallabels1 = vallabels1.data.cpu().numpy()
                    valoutput1 = valoutput1.data.cpu().numpy()
                    mse1 += np.sum((valoutput1 - vallabels1) ** 2)

                    predict2_y = torch.max(valoutput2, dim=1)[1]
                    acc_v += torch.eq(predict2_y, vallabels2.to(device)).sum().item()

                    valrunning_loss += valloss.item()
                    valrunning_loss1 += valloss1.item()
                    valrunning_loss2 += valloss2.item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)

            val1_mse = mse1 / val_num
            val1_rmse = val1_mse ** 0.5
            val2_accurate = acc_v / val_num
            if val1_rmse < best_rmse:
                best_rmse = val1_rmse
                torch.save(net.state_dict(), os.path.join(BASE, 'checkpoints', 'dual_inceptionv3', 'dual_inceptionv3_best.pth'))

            scheduler.step(val1_rmse)
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

            print(
                '[epoch %d] train_loss: %.3f train_loss1: %.3f train_loss2: %.3f vloss1: %.3f vloss2: %.3f train_acc: %.3f val_rmse: %.3f val2_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, running_loss1 / train_steps, running_loss2 / train_steps,
                 valrunning_loss1 / val_steps, valrunning_loss2 / val_steps,
                 train_acc,
                 val1_rmse, val2_accurate))
            f.write(
                "[Epoch {0:3d}] ,train_loss: {1:.3f},, train_loss1: {2:.3f},,train_loss2: {3:.3f},, vloss1: {4:.3f},,vloss2: {5:.3f},,train_acc: {6:.3f},, val_rmse: {7:.3f},, val2_accuracy: {8:.3f}".format(
                    epoch + 1, running_loss / train_steps, running_loss1 / train_steps, running_loss2 / train_steps,
                    valrunning_loss1 / val_steps, valrunning_loss2 / val_steps,
                    train_acc, val1_rmse, val2_accurate))
            f.write("\n ")
        torch.save(net.state_dict(), os.path.join(BASE, 'checkpoints', 'dual_inceptionv3', 'dual_inceptionv3.pth'))

        print('Finished Training')


if __name__ == '__main__':
    main()
