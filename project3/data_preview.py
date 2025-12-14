import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris

# 加载数据集（比如Iris）
iris = load_iris()

# 将 sklearn 的 Bunch 对象转换为 DataFrame
df = pd.DataFrame(iris.data, columns=[
    'sepal_length',  # 重命名为简单格式
    'sepal_width', 
    'petal_length', 
    'petal_width'
])
# 添加目标列
df['species'] = iris.target
df['species_name'] = iris.target_names[iris.target]
print(df.iloc[50:100])

# 数据预处理
df = df.dropna()  # 删除缺失值
df['species'] = df['species'].astype('category').cat.codes  # 将类别型数据转化为数值型

# 创建多个子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 创建2行2列的子图

# 第一个子图：sepal_length 的箱线图
sns.boxplot(x='species', y='sepal_length', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length by Species')

# 第二个子图：sepal_width 的箱线图
sns.boxplot(x='species', y='sepal_width', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width by Species')

# 第三个子图：petal_length 的箱线图
sns.boxplot(x='species', y='petal_length', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length by Species')

# 第四个子图：petal_width 的箱线图
sns.boxplot(x='species', y='petal_width', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Petal Width by Species')

# 调整布局，使得子图之间不重叠
plt.tight_layout()
plt.show()

# 使用Plotly绘制交互式散点图：绘制每一对特征之间的散点图
fig1 = px.scatter(df, x='sepal_length', y='sepal_width', color='species', title="Sepal Length vs Sepal Width")
fig2 = px.scatter(df, x='sepal_length', y='petal_length', color='species', title="Sepal Length vs Petal Length")
fig3 = px.scatter(df, x='sepal_length', y='petal_width', color='species', title="Sepal Length vs Petal Width")
fig4 = px.scatter(df, x='sepal_width', y='petal_length', color='species', title="Sepal Width vs Petal Length")
fig5 = px.scatter(df, x='sepal_width', y='petal_width', color='species', title="Sepal Width vs Petal Width")
fig6 = px.scatter(df, x='petal_length', y='petal_width', color='species', title="Petal Length vs Petal Width")

# 显示交互式图表
fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
fig6.show()
