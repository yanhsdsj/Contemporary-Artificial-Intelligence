import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_processing import ImageTextDataset
from model import MultiModalEmotionModel
from config import Config

# 加载配置
config = Config()

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

# 加载测试数据集
test_dataset = ImageTextDataset(image_dir=config.image_data_dir, text_file=config.test_file, tokenizer=tokenizer, mode='test')
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 初始化模型
model = MultiModalEmotionModel(
    pretrained_bert=config.pretrained_model,
    hidden_size=config.hidden_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers
)

# 加载训练好的模型权重
# model_path = os.path.join(config.test_output_dir, 'best_multimodal_model_20250120_134341.pth')
model_path = r"C:\Users\admin\Desktop\project\result\best_multimodal_model_20250120_174137.pth"
model.load_state_dict(torch.load(model_path, map_location=config.device))
model.to(config.device)
model.eval()

# 预测函数
def predict(model, data_loader, device):
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

# 进行预测
predictions = predict(model, test_loader, config.device)

# 将预测结果映射为标签
tag_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_tags = [tag_map[pred] for pred in predictions]

# 读取测试文件
test_data = pd.read_csv(config.test_file)

# 更新标签
test_data['tag'] = predicted_tags

# 保存预测结果
output_file = os.path.join(config.test_output_dir, 'predictions.txt')
test_data.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")