import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleGCN(torch.nn.Module):
    """
    简单的图卷积网络模型。
    """
    def __init__(self, num_node_features, hidden_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x

def train_model(num_epochs=1000, batch_size=1, learning_rate=0.01):
    # 加载之前保存的图数据，包括目标标签
    graph_data_list = torch.load('molecular_graphs_with_labels.pt')

    # 数据分割
    num_data = len(graph_data_list)
    num_train = int(num_data * 0.8)
    train_data_list = graph_data_list[:num_train]
    test_data_list = graph_data_list[num_train:]

    # 创建DataLoader
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    # 初始化模型
    num_node_features = train_data_list[0].num_node_features
    model = SimpleGCN(num_node_features=num_node_features, hidden_channels=64)

    # 定义损失函数和优化器
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_func(out.squeeze(), batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 测试模型
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_func(out.squeeze(), batch.y)
                total_test_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}, Test Loss: {total_test_loss / len(test_loader)}')

    # 保存训练好的模型
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    train_model()
