import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 加载数据
data1 = pd.read_parquet('train-data-1.parquet')
data2 = pd.read_parquet('train-data-2.parquet')
data3 = pd.read_parquet('test-validation.parquet')

# 合并训练数据
train_data = pd.concat([data1, data2], ignore_index=True)

# 移除包含无效标签值的行
train_data = train_data.dropna(subset=[train_data.columns[185]])  # 假设第186列是标签列

# 提取特征和标签
features = train_data.iloc[:, :185].values
labels = train_data.iloc[:, 185].values

# 标签标准化
scaler = StandardScaler()
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1)).ravel()

# 划分数据集
trainX, testX, trainy, testy = train_test_split(
    features, labels_scaled, test_size=0.2, random_state=42, shuffle=True
)

# 创建XGBoost回归模型
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 训练模型
xgb_model.fit(trainX, trainy, eval_metric="rmse")

# 预测三个数据集
data1['forecast'] = xgb_model.predict(data1.iloc[:, :185])
data2['forecast'] = xgb_model.predict(data2.iloc[:, :185])
data3['forecast'] = xgb_model.predict(data3.iloc[:, :185])

# 合并预测结果
forecast_data = pd.concat([data1[['time', 'asset_id', 'forecast']],
                           data2[['time', 'asset_id', 'forecast']],
                           data3[['time', 'asset_id', 'forecast']]], ignore_index=True)

# 保存预测结果到Parquet文件
forecast_data.to_parquet('forecast.parquet')


