# Курсовая работа
# Анализ поведенческих закономерностей пользователей социальных сетей
# Шуркалова А. Н.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

# 1. Загружаем данные

df = pd.read_csv('instagram_usage_lifestyle.csv')
print('Загружено', df.shape[0], 'строк,', df.shape[1], 'столбцов')

# Берём 50000 случайных пользователей
df = df.sample(n=50000, random_state=42)
print('Используем', len(df), 'записей')

# 2. Отбираем признаки

cols = [
    'daily_active_minutes_instagram',
    'likes_given_per_day',
    'comments_written_per_day',
    'dms_sent_per_week',
    'dms_received_per_week',
    'followers_count',
    'following_count',
    'sessions_per_day',
    'posts_created_per_week',
    'stories_viewed_per_day',
    'reels_watched_per_day',
    'ads_clicked_per_day'
]

X = df[cols].copy()
X = X.dropna()
print('Признаков:', len(cols))
print('Объектов после очистки:', len(X))

# 3. Стандартизация

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Метод локтя

inertia = []
for k in range(1, 11):
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    m.fit(X_scaled)
    inertia.append(m.inertia_)

plt.plot(range(1, 11), inertia, 'o-')
plt.xlabel('Число кластеров')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.grid()
plt.savefig('elbow.png')
plt.show()

# 5. K-Means с 3 кластерами

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

# 6. Размер кластеров

c = Counter(clusters)
print()
print('Распределение по кластерам:')
print('Кластер 0:', c[0], '(', round(c[0]/len(clusters)*100, 1), '%)')
print('Кластер 1:', c[1], '(', round(c[1]/len(clusters)*100, 1), '%)')
print('Кластер 2:', c[2], '(', round(c[2]/len(clusters)*100, 1), '%)')

# 7. Силуэт

sil = silhouette_score(X_scaled, clusters)
print()
print('Силуэт:', round(sil, 4))

# 8. Визуализация кластеров

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green']
names = ['Пассивные', 'Средние', 'Активные']
for i in range(3):
    mask = (clusters == i)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=names[i], alpha=0.5)
plt.legend()
plt.title('Кластеры пользователей')
plt.savefig('clusters.png')
plt.show()

# 9. Создаём метки для KNN

df2 = df[cols].copy()
df2['score'] = (df2['daily_active_minutes_instagram'] / 100 +
                df2['likes_given_per_day'] / 50 +
                df2['comments_written_per_day'] / 10 +
                df2['posts_created_per_week'] * 10 +
                df2['sessions_per_day'] * 5)

q1 = df2['score'].quantile(0.33)
q2 = df2['score'].quantile(0.66)

def get_label(s):
    if s <= q1:
        return 0
    elif s <= q2:
        return 1
    else:
        return 2

df2['label'] = df2['score'].apply(get_label)
labels = df2['label'].values

print()
print('Метки для KNN:')
print('Пассивные:', sum(labels == 0), '(', round(sum(labels == 0)/len(labels)*100, 1), '%)')
print('Средние:', sum(labels == 1), '(', round(sum(labels == 1)/len(labels)*100, 1), '%)')
print('Активные:', sum(labels == 2), '(', round(sum(labels == 2)/len(labels)*100, 1), '%)')

# 10. Обучаем KNN

X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42, stratify=labels)

print()
print('Обучающая выборка:', len(X_train))
print('Тестовая выборка:', len(X_test))
print()
print('Результаты KNN:')
print('k   | Accuracy')
print('----|---------')

best_acc = 0
best_k = 0

for k in [3, 5, 7, 9, 11, 15, 21]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(k, ' | ', round(acc, 4))
    if acc > best_acc:
        best_acc = acc
        best_k = k

print()
print('Лучшее k =', best_k, 'с точностью', round(best_acc*100, 2), '%')

# 11. Матрица ошибок

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
pred_best = best_knn.predict(X_test)

cm = confusion_matrix(y_test, pred_best)
print()
print('Матрица ошибок:')
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Пассивные', 'Средние', 'Активные'],
            yticklabels=['Пассивные', 'Средние', 'Активные'])
plt.xlabel('Предсказано')
plt.ylabel('На самом деле')
plt.title('Матрица ошибок KNN')
plt.savefig('confusion_matrix.png')
plt.show()

# 12. Сравнение метрик расстояния

print()
print('Сравнение метрик:')
print('Метрика       | Точность')
print('--------------|---------')

for m in ['euclidean', 'manhattan', 'chebyshev']:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=m)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(m, ' | ', round(acc, 4))

print()
print('Готово')
