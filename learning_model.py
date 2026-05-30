from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


# ====== Шаг 1: Загрузка и подготовка данных ======
# Загружаем датасет (например, Short Jokes Dataset или любой другой)
data = pd.read_csv("dataset.csv")
texts = data["text"]  # Тексты шуток
# Преобразуем TRUE/FALSE в 1/0
labels = data["humor"].map({True: 1, False: 0})  # Преобразуем TRUE/FALSE в 1/0

# Разделяем данные на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ====== Шаг 2: Токенизация ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

# Преобразуем метки в тензоры
train_labels = torch.tensor(list(train_labels.values))
test_labels = torch.tensor(list(test_labels.values))

# ====== Шаг 3: Создание Dataset и DataLoader ======
class HumorDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = HumorDataset(train_encodings, train_labels)
test_dataset = HumorDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ====== Шаг 4: Загрузка модели и настройка ======
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# # ====== Шаг 5: Тренировка модели ======
# model.train()

# # Параметры для ранней остановки
# early_stopping_patience = 3  # Количество эпох без улучшений перед остановкой
# best_loss = float("inf")  # Лучшее значение потерь на текущий момент
# patience_counter = 0  # Счётчик эпох без улучшений

# # Задаём минимально допустимое значение Loss
# min_acceptable_loss = 0.005  # Пример: остановить, если Loss меньше 0.1

# for epoch in range(10):  # Устанавливаем максимальное количество эпох
#     print(f"Epoch {epoch + 1}")
#     epoch_loss = 0

#     for step, batch in enumerate(train_loader):
#         optimizer.zero_grad()
#         batch = {key: val.to(device) for key, val in batch.items()}

#         # Прогоняем данные через модель
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#         # Суммируем Loss
#         epoch_loss += loss.item()

#         # Периодический вывод Loss
#         if step % 10 == 0:
#             print(f"Step {step}, Loss: {loss.item()}")

#         # Прерывание, если Loss ниже порога
#         if loss.item() < min_acceptable_loss:
#             print(f"Stopping early: Loss {loss.item()} < {min_acceptable_loss}")
#             break

#     # Средний Loss за эпоху
#     average_epoch_loss = epoch_loss / len(train_loader)
#     print(f"Average loss for epoch {epoch + 1}: {average_epoch_loss}")

#     # Прерывание, если средний Loss ниже порога
#     if average_epoch_loss < min_acceptable_loss:
#         print(f"Stopping early: Average loss {average_epoch_loss} < {min_acceptable_loss}")
#         break
# #+============================================================================================



#Загрузка модели++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Указываем устройство
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Загружаем сохранённое состояние
checkpoint = torch.load("trained_model.pth", map_location=device)

# Восстанавливаем веса модели и параметры оптимизатора
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]
print(f"Model loaded. Resuming from epoch {start_epoch + 1}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ====== Прогон модели по тестовым данным и сохранение результатов ======
output_file = "test_results.csv"  # Название файла для сохранения результатов

# Перевод модели в режим оценки
model.eval()

# Список для хранения результатов
results = []
real_labels = []  # Список для реальных меток
predicted_labels = []  # Список для предсказанных меток


TP = 0  # Истинные положительные
FP = 0  # Ложные положительные
FN = 0  # Ложные отрицательные
TN = 0  # Истинные отрицательные

total = 0  # Общее количество примеров
correct = 0  # Количество правильных предсказаний


# Предсказание
with torch.no_grad():
    for batch in test_loader:
        # Подготовка батча
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        # Прогон через модель
        outputs = model(**inputs)
        logits = outputs.logits

        # Преобразование логитов в вероятности
        probabilities = softmax(logits, dim=-1)

        # Предсказанные классы
        predictions = torch.argmax(logits, dim=-1)
        
        # Обновляем метрики
        TP += ((predictions == 1) & (labels == 1)).sum().item()  # Истинные положительные
        FP += ((predictions == 1) & (labels == 0)).sum().item()  # Ложные положительные
        FN += ((predictions == 0) & (labels == 1)).sum().item()  # Ложные отрицательные
        TN += ((predictions == 0) & (labels == 0)).sum().item()  # Истинные отрицательные
        
        total += labels.size(0)  # Общее количество элементов в батче
        correct += (predictions == labels).sum().item()  # Количество правильных предсказаний

        # Сохранение результатов
        for i in range(len(labels)):
            text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
            real_label = labels[i].item()
            predicted_label = predictions[i].item()
            prob_humorous = probabilities[i][1].item()
            prob_not_humorous = probabilities[i][0].item()
            results.append([text, real_label, predicted_label, prob_humorous, prob_not_humorous])

# Accuracy
accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

# Precision
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
print(f"Precision: {precision:.4f}")

# Recall
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
print(f"Recall: {recall:.4f}")

# F1-score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
print(f"F1-score: {f1:.4f}")


# Сохранение результатов в CSV файл
results_df = pd.DataFrame(results, columns=["Text", "Real Label", "Predicted Label", "Probability Humorous", "Probability Not Humorous"])
results_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Results saved to {output_file}")






# ====== Шаг 6: Предсказание на новом тексте ======
# # Новый текст для предсказания
# new_texts = ["Why don’t scientists trust atoms? Because they make up everything!",
#              "It's a sunny day."]

# # Токенизация нового текста
# new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
# # Перевод модели в режим оценки
# model.eval()

# # Предсказание
# with torch.no_grad():
#     new_encodings = {key: val.to(device) for key, val in new_encodings.items()}
#     outputs = model(**new_encodings)
#     logits = outputs.logits
#     # Вывод логитов
#     print("Logits:")
#     print(logits)
#     # Преобразование логитов в вероятности
#     probabilities = softmax(logits, dim=-1)
#     print("Probabilities:")
#     print(probabilities)
#     predictions = torch.argmax(logits, dim=-1)

# # Расшифровка результатов
# predicted_classes = ["Humorous" if pred == 1 else "Not Humorous" for pred in predictions]
# print("Predictions for new texts:")
# for text, label in zip(new_texts, predicted_classes):
#     print(f"'{text}' -> {label}")
#+++++++++++++++++++++++++++++++++++Чисто по 2 текстам прогон+++++++++++++++++++




#++++++++++++++++++++++++++++++++++++++Saving of model++++++++++++++++++++++++++
# # Сохранение модели
# save_path = "trained_model.pth"

# torch.save({
#     "model_state_dict": model.state_dict(),  # Веса модели
#     "optimizer_state_dict": optimizer.state_dict(),  # Параметры оптимизатора
#     "epoch": epoch,  # Последняя эпоха
# }, save_path)

# print(f"Model saved to {save_path}")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



