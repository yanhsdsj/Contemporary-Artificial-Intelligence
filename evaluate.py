import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_processing import ImageTextDataset
from model import MultiModalEmotionModel
from config import Config

config = Config()

tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

test_dataset = ImageTextDataset(image_dir=config.image_data_dir, text_file=config.test_file, tokenizer=tokenizer, mode='test')
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 初始化模型
model = MultiModalEmotionModel(
    pretrained_bert=config.pretrained_model,
    hidden_size=config.hidden_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers
)

model_path = os.path.join(config.test_output_dir, 'best_multimodal_model.pth')
model.load_state_dict(torch.load(model_path, map_location=config.device))
model.to(config.device)
model.eval()

def predict(model, data_loader, device):
    """
    模型预测
    """
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

predictions = predict(model, test_loader, config.device)

# 预测结果映射为标签
tag_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_tags = [tag_map[pred] for pred in predictions]

test_data = pd.read_csv(config.test_file)

test_data['tag'] = predicted_tags

output_file = os.path.join(config.test_output_dir, 'predictions.txt')
test_data.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")