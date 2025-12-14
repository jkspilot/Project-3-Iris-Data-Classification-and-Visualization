import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 加载数据
iris = load_iris()
X = iris.data[:, 2:]  # 使用后两个特征（花瓣长度和宽度）
y = iris.target
feature_names = ['Petal Length', 'Petal Width']
target_names = iris.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===================== 定义分类器 =====================

classifiers = {
    # 1. 逻辑回归（不同正则化强度）
    "Logistic (C=0.1)": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    "Logistic (C=1)": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Logistic (C=100)": LogisticRegression(C=100, max_iter=1000, random_state=42),
    
    # 2. 高斯过程
    "Gaussian Process": GaussianProcessClassifier(
        kernel=1.0 * RBF([1.0, 1.0]), 
        random_state=42
    ),
    
    # 3. 梯度提升
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    
    # 4. 逻辑回归 + RBF特征
    "Logistic + RBF features": make_pipeline(
        StandardScaler(),
        Nystroem(kernel="rbf", gamma=0.5, n_components=50, random_state=42),
        LogisticRegression(C=10, max_iter=1000, random_state=42),
    ),
    
    # 5. 逻辑回归 + 分箱特征
    "Logistic + Binned features": make_pipeline(
        StandardScaler(),
        KBinsDiscretizer(n_bins=5, encode='onehot-dense', 
                        strategy='quantile', random_state=42),
        PolynomialFeatures(interaction_only=True, include_bias=False),
        LogisticRegression(C=10, max_iter=1000, random_state=42),
    ),
    
    # 6. 逻辑回归 + 样条特征
    "Logistic + Spline features": make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=True),
        PolynomialFeatures(interaction_only=True, include_bias=False),
        LogisticRegression(C=10, max_iter=1000, random_state=42),
    ),
}

# 训练所有分类器
print("训练分类器中...")
models = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    models[name] = clf

# ===================== 创建可视化 =====================
# 设置图形
n_classifiers = len(classifiers)
n_cols = 4  # 每个分类器显示4个子图：class0, class1, class2, max class
n_rows = n_classifiers

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

# 生成网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# 定义颜色
class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # class 0, 1, 2 的颜色
max_class_cmap = plt.cm.colors.ListedColormap(['#FFB6B6', '#B6FFE8', '#B6D7FF'])  # 最大类别的颜色

print("开始绘制决策边界...")

# 为每个分类器绘制子图
for row_idx, (name, model) in enumerate(models.items()):
    print(f"正在绘制 {name}...")
    
    # 准备网格数据
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        # 预测概率
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(mesh_points)
            max_class = proba.argmax(axis=1)
        else:
            # 对于不支持概率预测的模型，使用预测结果
            predictions = model.predict(mesh_points)
            proba = np.zeros((len(mesh_points), 3))
            for i in range(3):
                proba[:, i] = (predictions == i).astype(float)
            max_class = predictions
        
        # 重塑为网格形状
        proba_grids = [proba[:, i].reshape(xx.shape) for i in range(3)]
        max_class_grid = max_class.reshape(xx.shape)
        
        # 绘制每个类别的概率和最大类别
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx < 3:  # class 0, 1, 2 的概率图
                # 绘制概率热图
                contour = ax.contourf(xx, yy, proba_grids[col_idx], 
                                     levels=20, cmap='viridis', alpha=0.8)
                
                # 绘制0.5等高线（决策边界）
                ax.contour(xx, yy, proba_grids[col_idx], 
                          levels=[0.5], colors='red', linewidths=2, alpha=0.8)
                
                # 绘制测试数据点
                for i in range(3):
                    mask = (y_test == i)
                    if mask.sum() > 0:
                        ax.scatter(X_test[mask, 0], X_test[mask, 1], 
                                  c=[class_colors[i]], edgecolors='k', s=30, 
                                  label=target_names[i] if col_idx == 0 and i == 0 else "", 
                                  alpha=0.8)
                
                ax.set_title(f'{name}\nClass {col_idx} ({target_names[col_idx]}) Probability', fontsize=10)
                
            else:  # 最大类别图
                # 绘制最大类别区域
                im = ax.imshow(max_class_grid, extent=(x_min, x_max, y_min, y_max), 
                              origin='lower', aspect='auto', 
                              cmap=max_class_cmap, alpha=0.6)
                
                # 绘制决策边界
                ax.contour(xx, yy, max_class_grid, colors='black', 
                          linewidths=1.5, levels=[0.5, 1.5], alpha=0.8)
                
                # 绘制测试数据点
                for i in range(3):
                    mask = (y_test == i)
                    if mask.sum() > 0:
                        ax.scatter(X_test[mask, 0], X_test[mask, 1], 
                                  c=[class_colors[i]], edgecolors='k', s=50, 
                                  label=target_names[i] if i == 0 else "", 
                                  alpha=1.0)
                
                ax.set_title(f'{name}\nMax Class Prediction', fontsize=10)
            
            # 设置轴标签
            if row_idx == n_rows - 1:  # 最后一行显示x轴标签
                ax.set_xlabel(feature_names[0])
            else:
                ax.set_xlabel('')
                
            if col_idx == 0:  # 第一列显示y轴标签
                ax.set_ylabel(feature_names[1])
            else:
                ax.set_ylabel('')
            
            # 设置坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 只在第一行第一列添加图例
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
                
    except Exception as e:
        print(f"绘制 {name} 时出错: {e}")
        # 如果出错，显示空白图
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].text(0.5, 0.5, f"Error\n{str(e)[:50]}", 
                                       ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
            axes[row_idx, col_idx].axis('off')

plt.tight_layout()
plt.suptitle('鸢尾花分类器可视化：各类别概率与最大类别预测', y=1.02, fontsize=16)
plt.show()