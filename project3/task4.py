import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
iris = load_iris()
X = iris.data[:, :3]  # 三个特征
y = iris.target  # 三个类别

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练SVM
clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
clf.fit(X_train_scaled, y_train)

# 预测
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# 创建3D图
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 定义鲜明颜色
colors = ['#FF0000', '#00FF00', '#0000FF']  # 红, 绿, 蓝

# 坐标范围
x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
z_min, z_max = X_test_scaled[:, 2].min() - 0.5, X_test_scaled[:, 2].max() + 0.5

# 创建三个平面来展示决策边界
resolution = 40

# 平面1：XY平面（固定z）
z_fixed = np.mean(X_test_scaled[:, 2])
xx1, yy1 = np.meshgrid(np.linspace(x_min, x_max, resolution),
                       np.linspace(y_min, y_max, resolution))

# 预测该平面上的类别
grid_points1 = np.c_[xx1.ravel(), yy1.ravel(), np.full(xx1.ravel().shape, z_fixed)]
grid_pred1 = clf.predict(grid_points1).reshape(xx1.shape)

# 绘制XY平面决策区域
for i in range(3):
    mask = grid_pred1 == i
    if mask.any():
        # 创建该类别区域
        z_surface = np.full_like(xx1, z_fixed)
        z_surface[~mask] = np.nan  # 只显示该类别
        
        ax.plot_surface(xx1, yy1, z_surface, 
                       color=colors[i], alpha=0.4, 
                       linewidth=0, antialiased=True)

# 平面2：XZ平面（固定y）
y_fixed = np.mean(X_test_scaled[:, 1])
xx2, zz2 = np.meshgrid(np.linspace(x_min, x_max, resolution),
                       np.linspace(z_min, z_max, resolution))

grid_points2 = np.c_[xx2.ravel(), np.full(xx2.ravel().shape, y_fixed), zz2.ravel()]
grid_pred2 = clf.predict(grid_points2).reshape(xx2.shape)

# 绘制XZ平面决策区域
for i in range(3):
    mask = grid_pred2 == i
    if mask.any():
        y_surface = np.full_like(xx2, y_fixed)
        y_surface[~mask] = np.nan
        
        ax.plot_surface(xx2, y_surface, zz2,
                       color=colors[i], alpha=0.3,
                       linewidth=0, antialiased=True)

# 绘制测试集数据点
for i in range(3):
    mask = (y_test == i)
    if mask.sum() > 0:
        # 正确分类的点
        correct_mask = mask & (y_pred == y_test)
        if correct_mask.sum() > 0:
            ax.scatter(X_test_scaled[correct_mask, 0], 
                      X_test_scaled[correct_mask, 1], 
                      X_test_scaled[correct_mask, 2],
                      c=colors[i], s=80, marker='o', 
                      edgecolors='k', alpha=0.9,
                      label=f'{iris.target_names[i]} (正确)')
        
        # 错误分类的点
        wrong_mask = mask & (y_pred != y_test)
        if wrong_mask.sum() > 0:
            ax.scatter(X_test_scaled[wrong_mask, 0], 
                      X_test_scaled[wrong_mask, 1], 
                      X_test_scaled[wrong_mask, 2],
                      c='black', s=100, marker='X', 
                      linewidths=2, alpha=1.0,
                      label='错误分类' if i == 0 else "")

# 设置图形属性
ax.set_xlabel('花萼长度', labelpad=12)
ax.set_ylabel('花萼宽度', labelpad=12)
ax.set_zlabel('花瓣长度', labelpad=12)

ax.set_title(f'鸢尾花三特征三分类3D决策边界\n'
             f'SVM (RBF核) | 准确率: {accuracy:.3f}\n'
             f'红色: {iris.target_names[0]}, 绿色: {iris.target_names[1]}, 蓝色: {iris.target_names[2]}', 
             fontsize=13, pad=20)

# 设置视角
ax.view_init(elev=25, azim=45)

# 添加图例
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()

print(f"测试准确率: {accuracy:.3f}")
print(f"正确分类: {np.sum(y_pred==y_test)}/{len(y_test)}")
print(f"错误分类: {np.sum(y_pred!=y_test)}")