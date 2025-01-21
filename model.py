import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class BERTEmotionModel(nn.Module):
    """
    基于预训练后的BERT模型提取文本的特征
    """
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_size=256, dropout_rate=0.3):
        super(BERTEmotionModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ResNetEmotionModel(nn.Module):
    """
    基于预训练后的ResNet模型提取图像的特征
    """
    def __init__(self, pretrained=True):
        super(ResNetEmotionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        x = self.resnet(images)
        x = self.dropout(x)
        return x
        
class MultiModalEmotionModel(nn.Module):
    """
    结合文本和图像特征(可选)，使用Transformer编码器和多头注意力机制进行多模态融合
    """
    def __init__(self, pretrained_bert='bert-base-uncased', hidden_size=256, num_heads=8, num_layers=6, use_text=True, use_image=True):
        super(MultiModalEmotionModel, self).__init__()
        self.use_text = use_text
        self.use_image = use_image

        if self.use_text:
            self.bert_model = BERTEmotionModel(pretrained_model_name=pretrained_bert, hidden_size=hidden_size)

        if self.use_image:
            self.resnet_model = ResNetEmotionModel(pretrained=True)
            
        if self.use_text and self.use_image:
            self.position_embeddings = nn.Embedding(5000, hidden_size)  # 假设最大序列长度为5000
            self.embedding = nn.Linear(256, hidden_size)  # 将图像特征映射到与文本特征相同的维度
            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=0.1)
            encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images, input_ids, attention_mask):
        if self.use_text:
            text_features = self.bert_model(input_ids, attention_mask)
            text_features = text_features.unsqueeze(0)

            if self.use_image:
                position_ids = torch.arange(0, text_features.size(1), device=text_features.device).unsqueeze(0)
                position_embeddings = self.position_embeddings(position_ids)
                text_features = text_features + position_embeddings
        else:
            text_features = None

        if self.use_image:
            image_features = self.resnet_model(images)
            if self.use_text:
                image_features = self.embedding(image_features).unsqueeze(0)
            else:
                image_features = image_features.unsqueeze(0)
        else:
            image_features = None

        if self.use_text and self.use_image:
            combined_features = torch.cat((text_features, image_features), dim=0)
            attn_output, _ = self.attention(combined_features, combined_features, combined_features)
            transformer_output = self.transformer_encoder(attn_output)
            final_feature = transformer_output.mean(dim=0)
        elif self.use_text:
            final_feature = text_features.squeeze(0)
        elif self.use_image:
            final_feature = image_features.squeeze(0)
        else:
            raise ValueError("At least one of use_text or use_image must be True")  # 简单错误处理
        output = self.fc(final_feature)
        return output
