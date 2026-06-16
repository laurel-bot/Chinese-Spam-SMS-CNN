import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import SpamDataset
from model import TextCNN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train():
    # Configurations
    data_path = 'data/spam_sms.txt' # You can place your mudou_spam.txt or target dataset here
    model_save_path = 'best_model.pth'
    vocab_save_path = 'vocab.pkl'
    
    # Check data file
    if not os.path.exists(data_path):
        # Create a tiny mock file if not exists for testing purposes
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write("0\t尊敬的旅客，您在南航官网订单支付成功，祝您旅途愉快。\n")
            f.write("1\t【天上掉下个Iphone5！】回对应数字设置优先接收的回执精品资讯，可参与抽奖。\n")
            f.write("0\t你好，今天晚上一起吃饭吗？\n")
            f.write("1\t特惠理财产品推荐，年化收益率高，快速到账，点击查看。\n")
            f.write("0\t工作总结已发送，请查收。\n")
            f.write("1\t恭喜您的号码已被抽中为一等奖，请速与我们联系。\n")
            f.write("0\t好的，我知道了，明天见。\n")
            f.write("1\t贷款利息低，额度高，无需抵押，点此链接即可申请。\n")
            print(f"Created a tiny sample dataset at {data_path}. Please replace it with the full mudou_spam dataset!")

    max_len = 100
    batch_size = 64
    epochs = 10
    embedding_dim = 128
    num_filters = 100
    filter_sizes = [3, 4, 5]
    learning_rate = 0.001
    
    # Load dataset
    dataset = SpamDataset(data_path, max_len=max_len, is_train=True)
    vocab = dataset.vocab
    
    # Save vocab
    with open(vocab_save_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    # Train / Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Handle tiny sample split
    if val_size == 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(len(vocab.word2idx), embedding_dim, num_filters, filter_sizes, num_classes=2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(targets.cpu().numpy())
                
        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} Prec: {val_precision:.4f} Rec: {val_recall:.4f} F1: {val_f1:.4f}")
              
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved best model with Val F1: {best_val_f1:.4f}")
            
    # Plotting training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Change Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training plot saved as 'training_curves.png'")

if __name__ == '__main__':
    train()
