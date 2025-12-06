import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
# Можно менять эти параметры
DATA_DIR = '/content/data'     # Путь к данным
MODEL_SAVE_PATH = 'plant_classifier_resnet18.pth'
PLOT_SAVE_PATH = 'training_plot.png'
CM_SAVE_PATH = 'confusion_matrix.png'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Устройство: {device}")
    if device.type == 'cpu':
        print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Вы используете CPU. Обучение может быть медленным.")
    return device

def prepare_data(data_dir, batch_size):
    # 1. Очистка от мусора (ipynb_checkpoints)
    checkpoint_folder = os.path.join(data_dir, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_folder):
        print(f"🧹 Удаление системной папки: {checkpoint_folder}")
        shutil.rmtree(checkpoint_folder)

    # 2. Трансформации (Аугментация)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("⏳ Загрузка датасета...")
    try:
        full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Папка {data_dir} не найдена.")
        exit(1)

    class_names = full_dataset.classes
    print(f"✅ Найдено классов: {len(class_names)}")
    
    # Разбиение 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Переопределяем transform для валидации (чтобы убрать аугментацию на тесте)
    # В ImageFolder transform применяется ко всему датасету, но random_split
    # создает Subset. Это упрощенный подход, для production лучше создать два ImageFolder.
    val_data.dataset.transform = val_transforms 

    dataloaders = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_data, batch_size=batch_size, shuffle=False)
    }
    
    return dataloaders, class_names, len(train_data), len(val_data)

def build_model(num_classes, device):
    print("🛠 Загрузка ResNet18...")
    model = models.resnet18(pretrained=True)
    
    # Замораживаем веса
    for param in model.parameters():
        param.requires_grad = False
    
    # Меняем голову
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    print(f"\n🏃 Начало обучения ({epochs} эпох)...")
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
    return history

def evaluate_and_save_reports(model, dataloaders, device, class_names):
    print("\n📊 Генерация отчетов...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Text Report
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH)
    print(f"💾 Матрица ошибок сохранена в '{CM_SAVE_PATH}'")
    plt.close()

def main():
    device = get_device()
    dataloaders, class_names, train_len, val_len = prepare_data(DATA_DIR, BATCH_SIZE)
    
    model = build_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Логика Checkpointing (Загружать или Обучать?)
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n💾 Найден файл модели: {MODEL_SAVE_PATH}")
        print("📥 Загрузка весов...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("✅ Модель загружена!")
        already_trained = True
    else:
        print("\n⚠️ Сохраненной модели нет. Запуск обучения...")
        history = train_model(model, dataloaders, criterion, optimizer, device, EPOCHS)
        
        # Сохранение весов
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Модель сохранена в '{MODEL_SAVE_PATH}'")
        
        # Сохранение графика обучения
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_SAVE_PATH)
        print(f"💾 График обучения сохранен в '{PLOT_SAVE_PATH}'")
        plt.close()
        already_trained = False

    # В любом случае проводим оценку
    evaluate_and_save_reports(model, dataloaders, device, class_names)

if __name__ == "__main__":
    main()