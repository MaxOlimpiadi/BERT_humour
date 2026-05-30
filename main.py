import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====== Шаг 1: Загрузка обученной модели ======
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Указываем устройство
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Загружаем сохранённое состояние
checkpoint = torch.load("trained_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()  # Переводим в режим оценки

# ====== Шаг 2: Загрузка тестового датасета ======
df = pd.read_excel("40000test.xlsx", header=None)  # Заголовков нет

texts = df[0]  # Первый столбец — тексты
true_labels = df[1]  # Второй столбец — реальные метки

# ====== Шаг 3: Токенизация ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=512, return_tensors="pt")

# ====== Шаг 4: Прогон через модель ======
predictions = []
probabilities = []

with torch.no_grad():
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = softmax(logits, dim=1)  # Применяем softmax
    preds = torch.argmax(probs, dim=1)  # Берём класс с наибольшей вероятностью

    predictions.extend(preds.cpu().numpy())
    probabilities.extend(probs.max(dim=1).values.cpu().numpy())  # Максимальная вероятность

# ====== Шаг 5: Вычисление метрик ======
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("Model Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

# ====== Шаг 6: Сохранение ошибок ======
errors_df = pd.DataFrame({
    "Text": texts,
    "Predicted Label": predictions,
    "Real Label": true_labels,
    "Confidence": probabilities
})

errors_df = errors_df[errors_df["Predicted Label"] != errors_df["Real Label"]]  # Фильтруем ошибки
errors_df.to_excel("errors.xlsx", index=False)
print(f"Ошибки сохранены в errors.xlsx (всего {len(errors_df)} ошибок)")

# ====== Шаг 7: Сохранение всех результатов ======
all_results_df = pd.DataFrame({
    "Text": texts,
    "Predicted Label": predictions,
    "Real Label": true_labels
})

all_results_df.to_excel("results_1000texts.xlsx", index=False)
print("Результаты всех 1000 текстов сохранены в results_40000texts.xlsx")
