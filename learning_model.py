from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime


# Training parameters:
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 10
LEARNING_RATE = 5e-5

# Model:
MODEL_NAME = "bert-base-uncased"
SAVE_BEST_PATH = 'best_model.pth'

# Datasets:
DATA_FOLDER = 'datasets'    
DATASET_FILE = 'kaggle_dataset.csv'

# Splits:
SPLIT_FOLDER = 'split/Kaggle_dataset_split'
TRAIN_SPLIT_FILE = 'train.csv'
VAL_SPLIT_FILE = 'val.csv'
TEST_SPLIT_FILE = 'test.csv'

# Test prediction results:
TEST_RESULTS_FOLDER = 'test'
TEST_RESULTS_FILE = 'predictions.csv'


# Experiment:
CREATE_SPLITS = False  # Если нужно датасет разбить на трейн, вал, тест. Иначе - сразу грузим все 3 части из соотв. файлов.
LOG_FILE_NAME = "experiments_log.csv"




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
    print(data["humor"].dtype)
    print(type(data["humor"].iloc[0]))
    texts = data["text"]  # Тексты шуток
    labels = data["humor"].astype(int)
    #labels = data["humor"].map({True: 1, False: 0}) # только для датасетов, где в поле humor лежать TRUE/FALSE
    
    return texts, labels



def create_splits(texts, labels):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, 
            labels,
            test_size = 0.2,
            shuffle = True,
            random_state = 42,
            stratify = labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, 
        temp_labels,
        test_size = 0.5,
        shuffle = True,
        random_state = 42,
        stratify = temp_labels
    )

    save_split(train_texts, train_labels, TRAIN_SPLIT_FILE)
    save_split(val_texts, val_labels, VAL_SPLIT_FILE)
    save_split(test_texts, test_labels, TEST_SPLIT_FILE)
    
    
    
def save_split(texts, labels, filename):
    df = pd.DataFrame(
        {
            'text': texts,
            'label': labels
        }    
    )
    full_path = os.path.join(SPLIT_FOLDER, filename)
    df.to_csv(full_path, index = False)
    


def load_split(filename):
    df = pd.read_csv(filename)  
    texts = df['text'] 
    labels = df['label']
    return texts, labels
        
    

def prepare_data(tokenizer):
    
    train_texts, train_labels = load_split(os.path.join(SPLIT_FOLDER, TRAIN_SPLIT_FILE))
    val_texts, val_labels = load_split(os.path.join(SPLIT_FOLDER, VAL_SPLIT_FILE))
    test_texts, test_labels = load_split(os.path.join(SPLIT_FOLDER, TEST_SPLIT_FILE))
    
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    
    # Преобразуем метки в тензоры
    train_labels = torch.tensor(train_labels.values)
    val_labels = torch.tensor(val_labels.values)
    test_labels = torch.tensor(test_labels.values)

    train_dataset = HumorDataset(train_encodings, train_labels)
    val_dataset = HumorDataset(val_encodings, val_labels)
    test_dataset = HumorDataset(test_encodings, test_labels)
        
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    
    return train_dataloader, val_dataloader, test_dataloader


def create_model(model_name = MODEL_NAME, lr = LEARNING_RATE):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    optimizer = AdamW(model.parameters(), lr = lr)
    
    return model, optimizer 


def train_model(model, optimizer, train_loader, val_dataloader, device):
    # Параметры для ранней остановки
    early_stopping_patience = 3  # Количество эпох без улучшений перед остановкой
    best_loss = float("inf")  # Лучшее значение потерь на текущий момент
    patience_counter = 0  # Счётчик эпох без улучшений
    
    for epoch in range(EPOCHS):  # Устанавливаем максимальное количество эпох
        model.train() # именно тут, потому что в конце при валиадции мы же в эвал модель переводим. Значит, надо обратно в трейн вернуть
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
    
        # Средний TRAIN LOSS за эпоху
        avg_train_epoch_loss = epoch_loss / len(train_loader)
        print(f"Average train loss for epoch {epoch + 1}: {avg_train_epoch_loss}")
        
        # Средний VAL LOSS за эпоху
        avg_val_epoch_loss = validate_model(model, val_dataloader, device)
        print(f"Average val loss for epoch {epoch + 1}: {avg_val_epoch_loss}")
        
        
        # Прерывание если средний val loss не улучшается на протяжении early_stopping_patience эпох:
        if avg_val_epoch_loss < best_loss:
            best_loss = avg_val_epoch_loss
            patience_counter = 0
            save_model(SAVE_BEST_PATH, model, optimizer)
        else:
            patience_counter += 1 
            if patience_counter >= early_stopping_patience:
                print(f'Stopping after {early_stopping_patience} epochs without val loss improvement')
                break
    



def validate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {
                key: val.to(device)
                for key, val in batch.items()
            }
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_dataloader)

    return avg_loss



def evaluate_model(model, test_loader, tokenizer, device):
    output_file_path = os.path.join(TEST_RESULTS_FOLDER, TEST_RESULTS_FILE)  # Название файла для сохранения результатов
    
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
    results_df.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Results saved to {output_file_path}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }   




def save_model(save_path, model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),  # Веса модели
            "optimizer_state_dict": optimizer.state_dict(),  # Параметры оптимизатора
        }, 
            save_path
    )
    
    print(f"Model saved to {save_path}")




def log_experiment(dataset_name, model_name, lr, batch_size, metrics_dict, log_filename = LOG_FILE_NAME):
    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "model": model_name,
        "lr": lr,
        "batch_size": batch_size,
        **metrics_dict # Распаковываем ваши TP, FP, F1 и т.д.
    }
    
    df = pd.DataFrame([row])
    
    # Если файл не существует, создаем его с колонками. Если существует — дописываем в конец (mode='a')
    header = not os.path.exists(log_filename)
    df.to_csv(log_filename, mode='a', index=False, header=header)




def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    if CREATE_SPLITS:
        data_path = os.path.join(DATA_FOLDER, DATASET_FILE)
        texts, labels = load_data(data_path)
        create_splits(texts, labels)
    
    train_dataset, val_dataset, test_dataset = prepare_data(tokenizer)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    model, optimizer = create_model()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    train_model(model, optimizer, train_dataloader, val_dataloader, device)
    
    checkpoint = torch.load(SAVE_BEST_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    metrics_dict = evaluate_model(model, test_dataloader, tokenizer, device)
    print(metrics_dict)
    
    log_experiment(DATASET_FILE, MODEL_NAME, LEARNING_RATE, BATCH_SIZE, metrics_dict)


    

    

if __name__ == "__main__":
    main()
    
    
    