import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_processing import ImageTextDataset
from model import MultiModalEmotionModel
from config import Config
from sklearn.metrics import accuracy_score, f1_score

# 加载配置
config = Config()

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

# 加载测试数据集
test_dataset = ImageTextDataset(image_dir=config.image_data_dir, text_file=config.test_file, tokenizer=tokenizer, mode='test')
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 初始化三个模型：仅文本、仅图像、多模态融合
text_only_model = MultiModalEmotionModel(
    pretrained_bert=config.pretrained_model,
    hidden_size=config.hidden_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    use_text=True,  # 仅使用文本
    use_image=False  # 不使用图像
)

image_only_model = MultiModalEmotionModel(
    pretrained_bert=config.pretrained_model,
    hidden_size=config.hidden_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    use_text=False,  # 不使用文本
    use_image=True  # 仅使用图像
)

multimodal_model = MultiModalEmotionModel(
    pretrained_bert=config.pretrained_model,
    hidden_size=config.hidden_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    use_text=True,  # 使用文本
    use_image=True  # 使用图像
)

# 加载训练好的模型权重
# text_only_model_path = os.path.join(config.test_output_dir, '\101\best_text_only_model_20250120_162559.pth')
# image_only_model_path = os.path.join(config.test_output_dir, '\101\best_image_only_model_20250120_160644.pth')
# multimodal_model_path = os.path.join(config.test_output_dir, '\101\best_multimodal_model_20250120_163949.pth')
text_only_model_path = r"C:\Users\admin\Desktop\project\result\101\best_text_only_model_20250120_162559.pth"
image_only_model_path = r"C:\Users\admin\Desktop\project\result\101\best_image_only_model_20250120_160644.pth"
multimodal_model_path = r"C:\Users\admin\Desktop\project\result\101\best_multimodal_model_20250120_163949.pth"

text_only_model.load_state_dict(torch.load(text_only_model_path, map_location=config.device))
image_only_model.load_state_dict(torch.load(image_only_model_path, map_location=config.device))
multimodal_model.load_state_dict(torch.load(multimodal_model_path, map_location=config.device))

# 将模型移动到设备
text_only_model.to(config.device)
image_only_model.to(config.device)
multimodal_model.to(config.device)

# 设置模型为评估模式
text_only_model.eval()
image_only_model.eval()
multimodal_model.eval()

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

# 分别进行预测
text_only_predictions = predict(text_only_model, test_loader, config.device)
image_only_predictions = predict(image_only_model, test_loader, config.device)
multimodal_predictions = predict(multimodal_model, test_loader, config.device)

# 将预测结果映射为标签
tag_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
text_only_tags = [tag_map[pred] for pred in text_only_predictions]
image_only_tags = [tag_map[pred] for pred in image_only_predictions]
multimodal_tags = [tag_map[pred] for pred in multimodal_predictions]

# 读取测试文件
test_data = pd.read_csv(config.test_file)

# 更新标签并保存结果
test_data['text_only_tag'] = text_only_tags
test_data['image_only_tag'] = image_only_tags
test_data['multimodal_tag'] = multimodal_tags

# 保存预测结果
output_file = os.path.join(config.test_output_dir, 'predictions_with_ablation.txt')
test_data.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")