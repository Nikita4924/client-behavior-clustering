
import pandas as pd
import pyarrow.parquet as pq
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns


print("Загружаю данные...")
table = pq.read_table("DECENTRATHON_3.0.parquet")
df = table.to_pandas()

required_columns = ['transaction_timestamp', 'card_id', 'transaction_amount_kzt', 'merchant_id', 'mcc_category']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Не хватает колонки: {col}")


df['timestamp'] = pd.to_datetime(df['transaction_timestamp'])


print("Вычисляю поведенческие признаки...")
features = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['mean', 'std', 'sum'],
    'merchant_id': 'nunique',
    'mcc_category': lambda x: x.mode().iloc[0] if not x.mode().empty else "NA",
    'timestamp': lambda x: (x.max() - x.min()).days
})

features.columns = [
    'amount_mean', 'amount_std', 'amount_sum',
    'unique_merchants', 'most_common_category',
    'activity_span_days'
]
features = features.reset_index()


print("Масштабирую и кластеризую...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features.select_dtypes(include=['float64', 'int64']))
clusterer = HDBSCAN(min_cluster_size=10)
features['cluster'] = clusterer.fit_predict(X_scaled)


print("Рисую визуализацию...")
embedding = umap.UMAP().fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=features['cluster'], palette='tab10')
plt.title('Кластеры клиентов')
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("clusters.png")
print("Визуализация сохранена: clusters.png")


features.to_parquet("client_segments.parquet", index=False)
features.to_csv("client_segments.csv", index=False)
print("Готово! Результаты сохранены.")
