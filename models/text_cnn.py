import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(
            self,
            num_filters: int,filter_sizes: list, vocab_size: int,
            embedding_size: int,sequence_length: int,num_classes: int,
            is_batch_normalize: bool = False, dropout_prob: float = None
        ):
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])
        self.batch_norm_list = nn.ModuleList([nn.BatchNorm2d(num_filters) for _ in filter_sizes]) if is_batch_normalize else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else None

    def forward(self, X):
        embedded_chars = self.W(X) 
        embedded_chars = embedded_chars.unsqueeze(1) 

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(embedded_chars))
            
            if self.batch_norm_list is not None:
                batch_norm = self.batch_norm_list[i]
                h = batch_norm(h)
            
            mp = nn.MaxPool2d((self.sequence_length - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)
            if self.dropout is not None:
                pooled = self.dropout(pooled)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) 
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) 
        model = self.Weight(h_pool_flat) + self.Bias 
        return model