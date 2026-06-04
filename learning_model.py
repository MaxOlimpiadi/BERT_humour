from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score



BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 10
LEARNING_RATE = 5e-5
MODEL_NAME = "bert-base-uncased"

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



def load_data(path):
    data = pd.read_csv(path)
    texts = data["text"]  # Тексты шуток
    labels = data["humor"].map({True: 1, False: 0})
    
    return texts, labels


def prepare_data(texts, labels, tokenizer):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    
    # Преобразуем метки в тензоры
    train_labels = torch.tensor(train_labels.values)
    test_labels = torch.tensor(test_labels.values)

    train_dataset = HumorDataset(train_encodings, train_labels)
    test_dataset = HumorDataset(test_encodings, test_labels)
        
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    
    return train_dataloader, test_dataloader


def create_model(model_name = MODEL_NAME, lr = LEARNING_RATE):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    optimizer = AdamW(model.parameters(), lr = lr)
    
    return model, optimizer 


def train_model(model, optimizer, train_loader, device):
    model.train()
    # Параметры для ранней остановки
    early_stopping_patience = 3  # Количество эпох без улучшений перед остановкой
    best_loss = float("inf")  # Лучшее значение потерь на текущий момент
    patience_counter = 0  # Счётчик эпох без улучшений
    
    # Задаём минимально допустимое значение Loss
    min_acceptable_loss = 0.005  # Пример: остановить, если Loss меньше 0.1
    
    for epoch in range(EPOCHS):  # Устанавливаем максимальное количество эпох
        print(f"Epoch {epoch + 1}")
        epoch_loss = 0
    
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {key: val.to(device) for key, val in batch.items()}
    
            # Прогоняем данные через модель
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
            # Суммируем Loss
            epoch_loss += loss.item()
    
            # Периодический вывод Loss
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
    
            # # Прерывание, если Loss ниже порога
            # if loss.item() < min_acceptable_loss:
            #     print(f"Stopping early: Loss {loss.item()} < {min_acceptable_loss}")
            #     break
    
        # Средний Loss за эпоху
        average_epoch_loss = epoch_loss / len(train_loader)
        print(f"Average loss for epoch {epoch + 1}: {average_epoch_loss}")
    
        # Прерывание, если средний Loss ниже порога
        if average_epoch_loss < min_acceptable_loss:
            print(f"Stopping early: Average loss {average_epoch_loss} < {min_acceptable_loss}")
            break
        
    return epoch




def evaluate_model(model, test_loader, tokenizer, device):
    output_file = "test_results.csv"  # Название файла для сохранения результатов
    
    # Перевод модели в режим оценки
    model.eval()
    
    # Список для хранения результатов
    results = []
    
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
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }   


def save_model(save_path, model, optimizer, epoch):
    torch.save(
        {
            "model_state_dict": model.state_dict(),  # Веса модели
            "optimizer_state_dict": optimizer.state_dict(),  # Параметры оптимизатора
            "epoch": epoch,  # Последняя эпоха
        }, 
            save_path
    )
    
    print(f"Model saved to {save_path}")


def main():
    data_path = 'dataset.csv'
    save_path = 'trained_model.pth'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    texts, labels = load_data(data_path)
    train_dataset, test_dataset = prepare_data(texts, labels, tokenizer)
    train_dataloader, test_dataloader = create_dataloaders(train_dataset, test_dataset)
    model, optimizer = create_model()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    last_epoch = train_model(model, optimizer, train_dataloader, device)
    metrics = evaluate_model(model, test_dataloader, tokenizer, device)
    print(metrics)

    save_model(save_path, model, optimizer, last_epoch)
    

    

if __name__ == "__main__":
    main()
    
    
    