import torch
import time
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from model import MultiModalEmotionModel
from data_processing import ImageTextDataset
from config import Config
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import os
from torch.utils.data import DataLoader

def train(config):
    """
    训练函数，可以选择使用什么模型（image_only/text_only/multi）进行训练
    """
    image_only_model = MultiModalEmotionModel(
        pretrained_bert=config.pretrained_model,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        use_text=False,
        use_image=True
    )
    text_only_model = MultiModalEmotionModel(
        pretrained_bert=config.pretrained_model,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        use_text=True,
        use_image=False
    )
    multimodal_model = MultiModalEmotionModel(
        pretrained_bert=config.pretrained_model,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        use_text=True,
        use_image=True
    )

    image_only_model.to(config.device)
    text_only_model.to(config.device)
    multimodal_model.to(config.device)

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    train_dataset = ImageTextDataset(config.image_data_dir, config.train_file, tokenizer, max_seq_length=config.max_seq_length, mode='train')

    train_size = int(config.split_rate * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # optimizer = AdamW(model.parameters(), lr=config.lr)
    # criterion = torch.nn.CrossEntropyLoss()

    image_only_optimizer = AdamW(image_only_model.parameters(), lr=config.lr, weight_decay=1e-5)
    text_only_optimizer = AdamW(text_only_model.parameters(), lr=config.lr, weight_decay=1e-5)
    multimodal_optimizer = AdamW(multimodal_model.parameters(), lr=config.lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    def train_model(model, optimizer, dataloader, device):
        """
        使用训练集训练模型
        """
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for batch in dataloader:
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tags = batch['tags'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_tags = torch.max(outputs, dim=1)
            running_accuracy += (predicted_tags == tags).sum().item()

        avg_loss = running_loss / len(dataloader)
        avg_accuracy = running_accuracy / len(train_dataset)
        return avg_loss, avg_accuracy

    def validate_model(model, dataloader, device):
        """
        对模型进行验证
        """
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                tags = batch['tags'].to(device)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, tags)

                val_loss += loss.item()
                _, predicted_tags = torch.max(outputs, dim=1)
                val_accuracy += (predicted_tags == tags).sum().item()

        val_loss = val_loss / len(dataloader)
        val_accuracy = val_accuracy / len(val_dataset)
        return val_loss, val_accuracy

    models = {
        "image_only": image_only_model,
        "text_only": text_only_model,
        "multimodal": multimodal_model
    }
    optimizers = {
        "image_only": image_only_optimizer,
        "text_only": text_only_optimizer,
        "multimodal": multimodal_optimizer
    }

    best_accuracy = {"image_only": 0.0, "text_only": 0.0, "multimodal": 0.0}
    best_model_paths = {"image_only": "", "text_only": "", "multimodal": ""}

    for epoch in range(config.epochs):
        # 一次性共训练三个模型
        print(f"Epoch {epoch + 1}/{config.epochs}")
        for model_name, model in models.items():
            print(f"Training {model_name} model...")
            train_loss, train_accuracy = train_model(model, optimizers[model_name], train_dataloader, config.device)
            val_loss, val_accuracy = validate_model(model, val_dataloader, config.device)

            print(f"{model_name} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"{model_name} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy[model_name]:
                best_accuracy[model_name] = val_accuracy
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                best_model_path = os.path.join(config.save_model_path_dir, f'best_{model_name}_model_{timestamp}.pth')
                torch.save(model.state_dict(), best_model_path)
                best_model_paths[model_name] = best_model_path
                print(f"Best {model_name} model saved at {best_model_path}")

    print("Training complete.")
    for model_name, path in best_model_paths.items():
        print(f"Best {model_name} model path: {path}")

if __name__ == "__main__":
    config = Config()
    train(config)
