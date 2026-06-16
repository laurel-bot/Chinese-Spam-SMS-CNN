import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embed = embed.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        pooled_outputs = []
        for conv in self.convs:
            # conv(embed) shape: [batch_size, num_filters, seq_len - kernel_size + 1]
            conved = F.relu(conv(embed))
            pooled = F.max_pool1d(conved, conved.shape[2]) # [batch_size, num_filters, 1]
            pooled_outputs.append(pooled.squeeze(2)) # [batch_size, num_filters]
            
        cat = self.dropout(torch.cat(pooled_outputs, dim=1)) # [batch_size, len(filter_sizes) * num_filters]
        logits = self.fc(cat)
        return logits
