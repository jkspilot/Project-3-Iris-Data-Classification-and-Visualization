from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, 2:]  # 选择后两个特征进行可视化
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 可视化决策边界
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1))

# 使用 predict_proba 方法获取每个点的概率
probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
probs = probs.reshape(xx.shape[0], xx.shape[1], 3)  # reshape to (height, width, classes)

# 设置每个类别的固定颜色
class_colors = ['yellow', 'green', 'blue']  # 自定义颜色：黄、绿、蓝

# 创建图形，画决策边界图 + 概率图
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# **1. 整体决策边界图**
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#  **使用imshow绘制决策区域**
# 使用 Z 来填充区域，确保每个类别的区域有不同的颜色
axs[0].imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin='lower',
          cmap=plt.cm.colors.ListedColormap(class_colors), alpha=0.6)
axs[0].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=mcolors.ListedColormap(class_colors))
axs[0].set_title('Overall Decision Boundaries')
axs[0].set_xlabel('Petal Length')
axs[0].set_ylabel('Sepal Width')

# **2, 3, 4. 每一类的概率图**
for i, class_prob in enumerate(probs.transpose(2, 0, 1)):  # i corresponds to each class
    ax = axs[i + 1]  # Use axs[1], axs[2], axs[3] for individual class probabilities
    
    # 对每个类别绘制概率图，并设置渐变色
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f'class_{i}_colormap', ['white', class_colors[i]], N=256)
    contour = ax.contourf(xx, yy, class_prob, alpha=0.7, cmap=cmap)
    
    # 画数据点，按照预测的类别显示
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=mcolors.ListedColormap(class_colors), alpha=1)
    
    # 添加color bar
    fig.colorbar(contour, ax=ax)
    
    # 设置标题和标签
    ax.set_title(f'Class {i} Probability')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Sepal Width')

# 调整布局
plt.tight_layout()
plt.show()

