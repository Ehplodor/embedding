import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        self.words = text.split()
        self.vocab = list(set(self.words))
        self.word2index = {word: i for i, word in enumerate(self.vocab)}
        self.index2word = {i: word for i, word in enumerate(self.vocab)}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        return self.word2index[word]

# Define the word embedding model
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout_rate):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        return embedded

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the dataset and dataloader
dataset = TextDataset('file.txt')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the model and optimizer
vocab_size = len(dataset.vocab)
embedding_dim = 200  # Recommended dimensionality of word embeddings
dropout_rate = 0.2  # Recommended dropout rate
model = WordEmbedding(vocab_size, embedding_dim, dropout_rate).to(device)
optimizer = Adam(model.parameters(), lr=0.001)  # Recommended learning rate

# Training loop
num_epochs = 10  # Recommended number of epochs
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        output = model(batch)
        loss = output.sum()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to avoid explosion
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Access the embedding matrix
embedding_matrix = model.embedding.weight.detach().cpu().numpy()

# Print the word embedding matrix
print(embedding_matrix)
