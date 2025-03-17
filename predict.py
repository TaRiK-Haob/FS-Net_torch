import torch
from model import FSNet
from train import MyDataset, DataInfo
from torch.utils.data import DataLoader
import os
from train import load_dataset

def predict(model, testloader, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    total_correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)

            loss, pred = model(x, y)
            total_loss += loss.item()

            # 检查 pred 的维度
            if pred.dim() > 1:  # 如果 pred 是二维张量
                total_correct += (pred.argmax(dim=1) == y).sum().item()
            else:  # 如果 pred 是一维张量
                total_correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(testloader)
    accuracy = total_correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def main():
    # 加载数据集
    batch_size = 64
    test_dataset = load_dataset(f'temp/test_dataset.jsonl')
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    max_train_len = 100
    # 加载模型
    model = FSNet(datainfo=DataInfo(65536 // 8, test_dataset.num_class, max_train_len))
    checkpoint_path = 'checkpoint/epoch_5000.pth'  # 修改为你的检查点路径

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # 执行预测
    predict(model, testloader, batch_size)

if __name__ == '__main__':
    main()