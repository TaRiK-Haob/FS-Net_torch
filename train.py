import torch, linecache, json
from model import FSNet
from torch.utils.data import DataLoader, Dataset, random_split
import os
from tqdm import tqdm

max_train_len = 100

class MyDataset(Dataset):
    def __init__(self, name):
        super(MyDataset, self).__init__()
        self.root = os.path.join('dataset', name+'.jsonl')

        with open(os.path.join('dataset', name+'.info'), 'r') as f:
            self.length = int(f.readline().strip())
            self.num_class = int(f.readline().strip())

    def _get_line(self, idx):
        try:
            line = linecache.getline(self.root, idx + 1).strip()
            if not line:  # 检查空行
                raise ValueError(f"Line {idx + 1} is empty")
            return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误在第{idx + 1}行: {e}")
            print(f"问题行内容: {line}")
            raise
        except Exception as e:
            print(f"读取第{idx + 1}行时出错: {e}")
            raise

    def __getitem__(self, index):
        json = self._get_line(index)
        pkt_len_seq = [abs(i) // 8 for i in json['pkt_len_seq']]
        if len(pkt_len_seq) > max_train_len:
            pkt_len_seq = pkt_len_seq[:max_train_len]
        else:
            pkt_len_seq = pkt_len_seq + [0] * (max_train_len - len(pkt_len_seq))
        pkt_len_seq = torch.tensor(pkt_len_seq)
        label = torch.tensor(int(json['label']))
        return pkt_len_seq, label

    def __len__(self):
        return self.length
    
def train(model, trainloader, valloader, checkpoint_path, batch_size, patience=10):
    print('Start training...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    epochs = 128
    lr = 0.0005
    batch_size = batch_size
    decay_rate = 0.5
    decay_step = len(trainloader.dataset) * 2 // batch_size + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    best_val_loss = float('inf')  # 初始化最小验证损失
    patience_counter = 0  # 早停计数器

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            loss, pred = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        scheduler.step()

        avg_train_loss = total_train_loss / len(trainloader)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in valloader:
                x = x.to(device)
                y = y.to(device)

                loss, pred = model(x, y)
                total_val_loss += loss.item()

                if pred.dim() > 1:
                    total_correct += (pred.argmax(dim=1) == y).sum().item()
                else:
                    total_correct += (pred == y).sum().item()
                total += y.size(0)

        avg_val_loss = total_val_loss / len(valloader)
        acc = total_correct / total
        tqdm.write(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best_model.pth'))
            tqdm.write(f"Best model saved at epoch {epoch}")
        else:
            patience_counter += 1
            tqdm.write(f"Early stopping patience counter: {patience_counter}/{patience}")

        # 检查是否需要早停
        if patience_counter >= patience:
            tqdm.write("Early stopping triggered")
            break

class DataInfo:
    def __init__(self, length_num, num_class, max_train_len):
        self.length_num = length_num
        self.num_class = num_class
        self.max_train_len = max_train_len

def save_dataset(dataset, file_path):
    """将数据集保存到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for pkt_len_seq, label in dataset:
            data = {
                'pkt_len_seq': pkt_len_seq.tolist(),
                'label': label.item()
            }
            f.write(json.dumps(data) + '\n')

def load_dataset(file_path):
    """从文件加载数据集"""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pkt_len_seq = torch.tensor(data['pkt_len_seq'])
            label = torch.tensor(data['label'])
            dataset.append((pkt_len_seq, label))
    return dataset

def main():
    dataset_dir = 'temp'
    train_file = os.path.join(dataset_dir, 'train_dataset.jsonl')
    test_file = os.path.join(dataset_dir, 'test_dataset.jsonl')
    val_file = os.path.join(dataset_dir, 'val_dataset.jsonl')

    batch_size = 128

    wholedataset = MyDataset('cicids2017')
    if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
        train_dataset = load_dataset(train_file)
        test_dataset = load_dataset(test_file)
        val_dataset = load_dataset(val_file)
        print('Dataset loaded')
    else:
        train_dataset, test_dataset, val_dataset = random_split(wholedataset, [0.7, 0.2, 0.1])
        save_dataset(train_dataset, 'temp/train_dataset.jsonl')
        save_dataset(test_dataset, 'temp/test_dataset.jsonl')
        save_dataset(val_dataset, 'temp/val_dataset.jsonl')
        print('Dataset saved & Loaded')

    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, prefetch_factor = 4)
    testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, prefetch_factor = 4)
    valloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, prefetch_factor = 4)

    max_pkt_len = 65536 // 8

    model = FSNet(datainfo=DataInfo(max_pkt_len, wholedataset.num_class, max_train_len))

    train(model, trainloader, valloader, 'checkpoint', batch_size)

if __name__ == '__main__':
    main()