import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
iris = load_iris()
# 使用前三个特征：花萼长度、花萼宽度、花瓣长度
X = iris.data[:, :3]
y = iris.target

# 转换为两分类问题（只取前两个类别）
binary_mask = y < 2
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 定义多个分类器 =====================

classifiers = {
    '逻辑回归': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'SVM (RBF核)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    'K-最近邻': KNeighborsClassifier(n_neighbors=5),
}

# ===================== 第一组图：3D决策边界 =====================
fig1 = plt.figure(figsize=(20, 15))

for idx, (name, clf) in enumerate(classifiers.items(), 1):
    # 训练模型
    clf.fit(X_train_scaled, y_train)
    
    # 创建3D子图
    ax = fig1.add_subplot(2, 2, idx, projection='3d')
    
    # 生成网格数据
    x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
    y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
    z_min, z_max = X_test_scaled[:, 2].min() - 0.5, X_test_scaled[:, 2].max() + 0.5
    
    # 创建网格用于绘制决策表面
    resolution = 50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # 固定z值为测试集z的平均值
    z_fixed = np.mean(X_test_scaled[:, 2])
    
    # 预测每个网格点的类别
    grid_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z_fixed)]
    grid_predictions = clf.predict(grid_points)
    Z = grid_predictions.reshape(xx.shape)
    
    # 绘制决策曲面
    surf = ax.plot_surface(xx, yy, Z, alpha=0.3, cmap='coolwarm', linewidth=0, antialiased=False)
    
    # 绘制测试集数据点
    colors = ['#FF6B6B', '#4ECDC4']
    y_pred = clf.predict(X_test_scaled)
    
    # 绘制正确分类的点
    for i in [0, 1]:
        correct_mask = (y_test == i) & (y_pred == y_test)
        if correct_mask.sum() > 0:
            ax.scatter(X_test_scaled[correct_mask, 0], 
                      X_test_scaled[correct_mask, 1], 
                      X_test_scaled[correct_mask, 2],
                      c=colors[i], s=80, marker='o', edgecolors='k', alpha=0.9)
    
    # 绘制错误分类的点
    wrong_mask = (y_pred != y_test)
    if wrong_mask.sum() > 0:
        ax.scatter(X_test_scaled[wrong_mask, 0], 
                  X_test_scaled[wrong_mask, 1], 
                  X_test_scaled[wrong_mask, 2],
                  c='black', s=100, marker='X', linewidths=2, alpha=1.0)
    
    # 设置图形属性
    ax.set_xlabel('花萼长度', labelpad=10)
    ax.set_ylabel('花萼宽度', labelpad=10)
    ax.set_zlabel('花瓣长度', labelpad=10)
    ax.set_title(f'\n{name}\n3D边界图', fontsize=12, pad=15)
    
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 添加图例
    if idx == 1:
        ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.suptitle('\n鸢尾花两分类3D决策边界可视化', y=1.02, fontsize=16)
plt.show()

# ===================== 第二组图：3D Probability Map  =====================
fig2 = plt.figure(figsize=(20, 15))

for idx, (name, clf) in enumerate(classifiers.items(), 1):
    # 创建3D子图
    ax = fig2.add_subplot(2, 2, idx, projection='3d')
    
    # 生成概率网格数据
    x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
    y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
    z_min, z_max = X_test_scaled[:, 2].min() - 0.5, X_test_scaled[:, 2].max() + 0.5
    
    # 创建2D网格用于概率曲面（x-y平面）
    resolution = 40
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # 固定z值为测试集z的平均值
    z_fixed = np.mean(X_test_scaled[:, 2])
    
    try:
        # 预测每个网格点的类别概率
        grid_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z_fixed)]
        
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(grid_points)
            # 类别1的概率（class 1 = versicolor）
            prob_class1 = proba[:, 1].reshape(xx.shape)
        else:
            raise ValueError(f"{name} 不支持概率预测")
        
        # ========== 绘制3D概率曲面 ==========
        # 将概率值作为z轴高度
        prob_surface_z = prob_class1 * (z_max - z_min) + z_min
        
        # 绘制概率曲面
        surf = ax.plot_surface(xx, yy, prob_surface_z, 
                              cmap='RdYlBu_r', alpha=0.8, 
                              linewidth=0.5, antialiased=True,
                              edgecolor='black')
        
        # 添加colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Class 1 (Versicolor) 概率', fontsize=10)
        
        # ========== 绘制曲面上的等高线 ==========
        # 在曲面上绘制等高线
        contour_levels = [0.3, 0.5, 0.7]
        for level in contour_levels:
            z_level = level * (z_max - z_min) + z_min
            # 找到该概率等级的轮廓
            cs = ax.contour(xx, yy, prob_class1, levels=[level], 
                          colors=['yellow', 'orange', 'red'][contour_levels.index(level)], 
                          linewidths=2, linestyles=['--', '-', '-.'][contour_levels.index(level)])
            
            # 将轮廓投影到曲面上
            for collection in cs.collections:
                collection.set_zorder(10)
        
        # ========== 绘制三个平面上的投影 ==========
        # 1. XY平面投影（底部）
        ax.contourf(xx, yy, prob_class1, zdir='z', offset=z_min-0.3, 
                   levels=20, cmap='RdYlBu_r', alpha=0.5)
        
        # 2. YZ平面投影（侧面）
        # 创建YZ网格
        yz_resolution = 30
        yy_yz, zz_yz = np.meshgrid(np.linspace(y_min, y_max, yz_resolution),
                                   np.linspace(z_min, z_max, yz_resolution))
        
        # 固定x为平均值
        x_fixed = np.mean(X_test_scaled[:, 0])
        grid_points_yz = np.c_[np.full(yy_yz.ravel().shape, x_fixed),
                               yy_yz.ravel(), zz_yz.ravel()]
        
        proba_yz = clf.predict_proba(grid_points_yz)
        prob_yz = proba_yz[:, 1].reshape(yy_yz.shape)
        
        ax.contourf(prob_yz, yy_yz, zz_yz, zdir='x', offset=x_min-0.3, 
                   levels=20, cmap='RdYlBu_r', alpha=0.5)
        
        # 3. XZ平面投影（另一个侧面）
        # 创建XZ网格
        xz_resolution = 30
        xx_xz, zz_xz = np.meshgrid(np.linspace(x_min, x_max, xz_resolution),
                                   np.linspace(z_min, z_max, xz_resolution))
        
        # 固定y为平均值
        y_fixed = np.mean(X_test_scaled[:, 1])
        grid_points_xz = np.c_[xx_xz.ravel(), np.full(xx_xz.ravel().shape, y_fixed),
                               zz_xz.ravel()]
        
        proba_xz = clf.predict_proba(grid_points_xz)
        prob_xz = proba_xz[:, 1].reshape(xx_xz.shape)
        
        ax.contourf(xx_xz, prob_xz, zz_xz, zdir='y', offset=y_min-0.3, 
                   levels=20, cmap='RdYlBu_r', alpha=0.5)
        
        # 添加等高线图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='yellow', linestyle='--', linewidth=2, label='P=0.3'),
            Line2D([0], [0], color='orange', linestyle='-', linewidth=2, label='P=0.5'),
            Line2D([0], [0], color='red', linestyle='-.', linewidth=2, label='P=0.7'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
    except Exception as e:
        print(f"{name} 概率图生成失败: {e}")
        ax.text2D(0.05, 0.95, f"不支持概率预测", transform=ax.transAxes, fontsize=10)
    
    # 绘制测试集数据点（真实标签）
    colors = ['#FF6B6B', '#4ECDC4']
    for i in [0, 1]:
        mask = (y_test == i)
        if mask.sum() > 0:
            # 计算该点在曲面上的投影z值
            points_for_projection = X_test_scaled[mask]
            if hasattr(clf, 'predict_proba'):
                proba_points = clf.predict_proba(points_for_projection)
                prob_values = proba_points[:, 1]
                z_projected = prob_values * (z_max - z_min) + z_min
                
                # 绘制点在曲面上的投影
                ax.scatter(points_for_projection[:, 0], 
                          points_for_projection[:, 1], 
                          z_projected,
                          c=colors[i], s=40, marker='o', edgecolors='k', alpha=0.7)
            
            # 绘制原始数据点
            ax.scatter(X_test_scaled[mask, 0], 
                      X_test_scaled[mask, 1], 
                      X_test_scaled[mask, 2],
                      c=colors[i], s=30, marker='^', edgecolors='k', alpha=0.5)
    
    # 设置图形属性
    ax.set_xlabel('花萼长度', labelpad=10)
    ax.set_ylabel('花萼宽度', labelpad=10)
    ax.set_zlabel('花瓣长度 / 概率', labelpad=10)
    ax.set_title(f'\n{name}\n3D概率曲面与投影', fontsize=12, pad=15)
    
    # 设置坐标轴范围
    ax.set_xlim([x_min-0.3, x_max])
    ax.set_ylim([y_min-0.3, y_max])
    ax.set_zlim([z_min-0.3, z_max+0.5])
    
    # 设置视角
    ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.suptitle('\n鸢尾花两分类3D概率图', y=1.02, fontsize=16)
plt.show()