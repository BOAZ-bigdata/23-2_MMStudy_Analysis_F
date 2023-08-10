from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model,inputs,targets,epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(inputs)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    
        loss.backward()
        optimizer.step()

def train_model_with_early_stop(model,inputs,targets,epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    patience = 5  
    counter = 0 
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, targets)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1} due to lack of improvement.')
                break
        loss.backward()
        optimizer.step()

