import torch
from torch_geometric.loader import DataLoader
from train_gcn import SimpleGCN
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 加载图数据列表
graph_data_list = torch.load('./molecular_graphs_with_labels.pt')

# 数据分割
num_data = len(graph_data_list)
num_train = int(num_data * 0.8)
test_data_list = graph_data_list[num_train:]

# 初始化模型
model = SimpleGCN(num_node_features=test_data_list[0].num_node_features, hidden_channels=64)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()  # 确保模型处于评估模式

# 创建 DataLoader，加载测试数据
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

# 收集预测和实际标签
predictions = []
actuals = []

# 禁用梯度计算
with torch.no_grad():
    for data in test_loader:
        # 前向传播
        out = model(data.x, data.edge_index, data.batch)
        # 存储预测结果
        predictions.extend(out.view(-1).tolist())
        # 存储实际结果
        actuals.extend(data.y.view(-1).tolist())

# 计算性能指标
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

# 输出性能指标
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R^2 Score: {r2}')

# 绘制对角线图
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.6)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red')  # y=x 对角线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.text(min(actuals), max(predictions), f'MAE: {mae:.2f}\nR^2: {r2:.2f}', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()

