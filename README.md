
---

## 🧰 Қолданылған технологиялар

- Python 3
- Pandas, NumPy – деректерді өңдеу
- Scikit-learn – ML модельдері мен метрикалар
- Matplotlib, Seaborn – визуализация
- Joblib – модельді сақтау

---

## ⚡ Функционал

1. Деректерді жүктеу және алдын-ала өңдеу  
2. Мақсатты бағананы (`target`) құру  
3. Негізгі сандық белгілерді таңдау  
4. Деректерді оқыту/тест бөлу  
5. Gaussian Naive Bayes моделі арқылы оқыту  
6. Модельді бағалау: accuracy, ROC-AUC, confusion matrix  
7. Модельді сақтау (`models/credit_model.pkl`)  
8. Нәтижелерді график ретінде көрсету:  
   - Probability distribution  
   - ROC curve  
   - Confusion matrix  

---

## 📈 Нәтижелер

- **Accuracy**: 0.XXXX  
- **ROC-AUC**: 0.XXXX  
- Confusion matrix, probability distribution және ROC curve `results/` папкасында сақталады.

---

## 🚀 Қолдану жолы

1. Jupyter Notebook немесе Python ортасын ашыңыз  
2. `bank.csv` файлын проект папкасына салыңыз  
3. `credit_scoring.ipynb` файлын іске қосыңыз  
4. Графиктер Notebook ішінде көрсетіледі және `results/` папкасына сақталады  
5. Модель `models/credit_model.pkl` файлына жазылады

---

## 💡 Ескерту

- CSV файлындағы сепаратор автоматты түрде анықталады (`;` немесе `,`)  
- Егер негізгі бағандар (`age`, `balance`, `duration`, `campaign`) жоқ болса, жүйе бірінші 4 сандық бағанды қолданады  

---

## 📬 Байланыс

Автор: **Temir Nurmakhan**  
Электрондық пошта: your_email@example.com
