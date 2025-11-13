import torch
from torch import nn
from torch.utils.data import DataLoader

from loss import get_wc, per_pixel_weight, dice_score
from model import u_net
from utils import ISBI2012, kfold, get_aug, init_weights, crop, accuracy_recall_precision


def train_val(D=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = kfold(D)
    train_transform, val_transform = get_aug()

    best_dice = 0.0
    best_state = None
    for fold in range(len(folds)):
        print(f"=====Fold:{fold}=====")
        model = u_net().to(device)
        model.apply(init_weights)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.99)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        wc = get_wc().to(device)

        train_slices, val_slices = folds[fold]
        train_ds = ISBI2012(slices=train_slices, is_train=True, transform=train_transform)
        val_ds = ISBI2012(slices=val_slices, is_train=True, transform=val_transform)

        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=1)
        model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        epochs = 60

        patience = 8  # 容忍连续多少轮没有提升
        best_loss = float('inf')  # 目前为止最好的（最小的）loss
        no_improve = 0  # 连续未提升轮数计数器
        best_state = None  # 最优模型参数

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_acc = 0.0
            total_recall = 0.0
            total_precision = 0.0
            for step, (x, y) in enumerate(train_dl, 1):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y = crop(y, tar_shape=y_hat.shape)
                acc, recall, precision = accuracy_recall_precision(y_hat, y)

                loss_map = criterion(y_hat, y).to(device)
                w = per_pixel_weight(y, wc)
                w = w.to(device)
                loss = (loss_map * w).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += acc
                total_recall += recall
                total_precision += precision

            val_loss, val_dice, avg_acc, avg_recall, avg_precision = evaluate(criterion, model, val_dl, wc, device)
            if val_dice > best_dice:
                best_dice = val_dice
                best_state = model.state_dict().copy()
            scheduler.step(val_loss)

            print(f"Epoch {epoch} \nval_loss={val_loss:.4f}  \nval_dice={val_dice:.4f}\n"
                  f"acc={avg_acc:.4f} recall={avg_recall:.4f} precision={avg_precision:.4f}")

        torch.save(best_state, f"unet_isbi2012_best_{fold}.pth")


def evaluate(criterion, model, val_dl, wc, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    total_recall = 0.0
    total_precision = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            y = crop(y, y_hat.shape)

            loss_map = criterion(y_hat, y).to(device)
            w = per_pixel_weight(y, wc)
            w = w.to(device)
            loss = (loss_map * w).mean()

            dice = dice_score(y_hat, y)

            total_loss += loss.item()
            total_dice += dice
            acc, recall, precision = accuracy_recall_precision(y_hat, y)
            total_acc += acc
            total_recall += recall
            total_precision += precision
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    avg_acc = total_acc / n_batches
    avg_recall = total_recall / n_batches
    avg_precision = total_precision / n_batches
    return avg_loss, avg_dice, avg_acc, avg_recall, avg_precision


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transform, val_transform = get_aug()

    model = u_net().to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    wc = get_wc().to(device)

    train_ds = ISBI2012(is_train=True, transform=train_transform)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    epochs = 60

    patience = 8  # 容忍连续多少轮没有提升
    best_loss = float('inf')  # 目前为止最好的（最小的）loss
    no_improve = 0  # 连续未提升轮数计数器
    best_state = None

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, (x, y) in enumerate(train_dl, 1):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y = crop(y, tar_shape=y_hat.shape)

            acc, recall, precision = accuracy_recall_precision(y_hat, y)
            loss_map = criterion(y_hat, y).to(device)
            w = per_pixel_weight(y, wc)
            w = w.to(device)

            loss = (loss_map * w - 0.1 * recall - 0.1 * precision).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        scheduler.step(avg_loss)

        delta = 1e-4
        if avg_loss < best_loss - delta:
            best_loss = avg_loss
            no_improve = 0
            best_state = model.state_dict().copy()
            print(f"    ✓ loss 改善，更新 best_loss={best_loss:.4f}")
        else:
            no_improve += 1
            print(f"    ✗ {no_improve} 轮没有变好 (patience={patience})")

        if no_improve >= patience:
            print(f"早停触发：连续 {patience} 轮无改进，停止训练。")
            break

        print(f"==> Epoch {epoch} \navg_loss={total_loss / len(train_dl):.4f}\n")

    torch.save(best_state, f"unet_isbi2012_wc.pth")
    print("模型已保存")


if __name__ == "__main__":
    # train_val()
    train()
