import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def run_defense_demo(csv_path='bank.csv'):
    print("🚀 Инициализация системы кредитного скоринга (Nagibaev Style)...")
    
    try:
        # 1. Автоопределение разделителя
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        sep = ';' if ';' in first_line else ','
        
        df = pd.read_csv(csv_path, sep=sep)
        print(f"✅ Данные загружены. Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
        
        # 2. Адаптация под кредит (Target: 1 - риск дефолта, 0 - надежный)
        if 'y' in df.columns:
            df['target'] = df['y'].map({'yes': 1, 'no': 0})
        elif 'deposit' in df.columns:
            df['target'] = df['deposit'].map({'yes': 1, 'no': 0})
        else:
            # Если нет явного таргета, берем последний столбец
            last_col = df.columns[-1]
            df['target'] = LabelEncoder().fit_transform(df[last_col])

        # 3. Выбор признаков (только числовые для простоты демо)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols: numeric_cols.remove('target')
        
        # Берем ключевые: age, balance, duration, campaign, day
        features = [c for c in ['age', 'balance', 'duration', 'campaign'] if c in numeric_cols]
        if not features: features = numeric_cols[:4]

        X = df[features].fillna(0)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Байесовское моделирование
        print(f"🧠 Обучение Gaussian Naive Bayes на признаках: {features}...")
        model = GaussianNB()
        model.fit(X_train, y_train)
        time.sleep(1)

        # 5. Интерактивная демонстрация
        print("\n" + "="*75)
        print(f"{'ID':<5} | {'Возраст':<8} | {'Баланс':<10} | {'P(Дефолт)':<12} | {'Вердикт'}")
        print("-" * 75)

        samples = X_test.head(15)
        probs = model.predict_proba(samples)[:, 1]

        for i, (idx, row) in enumerate(samples.iterrows()):
            prob = probs[i]
            # Устанавливаем порог риска 0.4 для строгости
            verdict = "❌ ОТКАЗ" if prob > 0.4 else "✅ ОДОБРЕНО"
            
            age = int(row.get('age', 0))
            balance = int(row.get('balance', 0))
            
            print(f"{i+1:<5} | {age:<8} | {balance:<10} | {prob:.2%}      | {verdict}")
            time.sleep(0.4) # Задержка для эффекта "думающей" системы

        print("="*75)
        print("\n📊 Анализ рисков завершен. Генерация отчета...")
        
        # Построение графика
        plt.figure(figsize=(10, 5))
        plt.hist(probs, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        plt.axvline(0.4, color='red', linestyle='--', label='Порог отсечения (Risk Threshold)')
        plt.title('Байесовское распределение вероятностей невозврата кредита')
        plt.xlabel('Вероятность дефолта')
        plt.ylabel('Количество заявок')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.savefig('scoring_results.png')
        print("📈 График распределения сохранен в 'scoring_results.png'")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    run_defense_demo()
