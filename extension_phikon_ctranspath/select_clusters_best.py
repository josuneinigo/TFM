import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from model.resnet import resnet50
from utils import MyDatasetCluster

def main():
    device = torch.device("cuda")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    BASE = '/home/jinigo/HSNP_code'
    cluster = sys.argv[1] if len(sys.argv) > 1 else '00'
    labels_dir = sys.argv[2] if len(sys.argv) > 2 else 'K=8'
    prefix = '' if labels_dir == 'K=8' else f'{labels_dir}_'
    txt_path1 = os.path.join(BASE, 'data', 'labels', labels_dir, f'{cluster}.txt')
    train_dataset = MyDatasetCluster(txt_path1, transform=data_transform["train"])
    train_num = len(train_dataset)
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    txt_path2 = os.path.join(BASE, 'data', 'labels', labels_dir, f'{cluster}_val.txt')
    validate_dataset = MyDatasetCluster(txt_path2, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    net = resnet50(num_classes=2)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    epochs = 20
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    best_val_acc = 0.0
    best_epoch = 0
    with open(f"./reporting/resnet_classification/{prefix}{cluster}.txt", "w") as f:
        for epoch in range(epochs):
            # train
            net.train()
            running_train_loss = 0.0
            acc_t = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                train_y = torch.max(logits, dim=1)[1]
                acc_t += torch.eq(train_y, labels.to(device)).sum().item()
                running_train_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
            train_acc = acc_t / train_num
            train_loss_avg = running_train_loss / train_steps
            # validate
            net.eval()
            acc_v = 0.0
            running_val_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(validate_loader, colour='green')
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    val_outputs = net(val_images.to(device))
                    loss1 = loss_function(val_outputs, val_labels.to(device))
                    predict_y = torch.max(val_outputs, dim=1)[1]
                    acc_v += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    running_val_loss += loss1.item()
                    val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss1)
            val_acc = acc_v / val_num
            val_loss_avg = running_val_loss / val_steps
            scheduler.step(val_loss_avg)
            current_lr = optimizer.param_groups[0]['lr']
            print(f",  Current learning rate is: {current_lr}")
            log_line = (f"[epoch {epoch+1}] train_loss: {train_loss_avg:.3f}  "
                        f"val_loss: {val_loss_avg:.3f}  train_accuracy: {train_acc:.3f} "
                        f"val_accuracy: {val_acc:.3f}")
            print(log_line)
            f.write(log_line + "\n")
            f.flush()
            # Guardar el MEJOR checkpoint visto hasta ahora (no solo el de la última época)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(BASE, 'checkpoints', 'resnet_classification', f'{prefix}{cluster}_resnet50_best.pth'))
        print(f'Finished Training. Best val_accuracy: {best_val_acc:.3f} (epoch {best_epoch})')
        f.write(f'BEST: val_accuracy={best_val_acc:.3f} at epoch {best_epoch}\n')

if __name__ == '__main__':
    main()
