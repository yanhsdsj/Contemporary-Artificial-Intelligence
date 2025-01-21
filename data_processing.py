import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from PIL import Image
import pandas as pd
from torchvision import transforms
from config import Config
import random
from nltk.corpus import wordnet
config = Config()
# 使用config中的路径
image_data_dir = config.image_data_dir
train_file = config.train_file
test_file = config.test_file


class ImageTextDataset(Dataset):
    """
    预先对数据集的处理
    """
    def __init__(self, image_dir, text_file, tokenizer, max_seq_length=512, mode='train'):
        """
        初始化数据集

        :param image_dir: 图像文件夹路径
        :param text_file: 文本文件路径
        :param tokenizer: 用于文本处理的BERT tokenizer
        :param max_length: 最大文本长度
        """
        self.image_dir = image_dir
        self.text_file = text_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.tag_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.data = pd.read_csv(text_file, delimiter=",", header=0, names=["guid", "tag"])
        # print("Loaded data:")
        # print(self.data.head())
        invalid_guids = self.data[self.data['guid'].isna() | (self.data['guid'] == 'guid')] # 检查 guid 是否有效
        if not invalid_guids.empty:
            print("Warning: Found invalid guids:")
            print(invalid_guids)

        self.image_paths = [os.path.join(image_dir, f"{guid}.jpg") for guid in self.data['guid']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据给定的索引idx从数据集中获取样本，
        对文本和图像进行处理后返回包含图像、文本编码和标签的字典
        """
        guid = self.data.iloc[idx, 0]
        tag = self.data.iloc[idx, 1]

        if pd.isna(tag):
            # print(f"Warning: NaN tag at index {idx}, replacing with default tag (0)")
            tag = 0

        if tag not in self.tag_map:
            # print(f"Warning: Invalid tag '{tag}' at index {idx}, replacing with default tag (0)")
            tag = 0  # 设置为默认标签，例如：'negative'
        else:
            tag = self.tag_map[tag]

        assert tag in [0, 1, 2], f"Invalid tag: {tag} at index {idx}"

        # 加载文本、图像；对文本进行BERT的tokenizer处理
        text = self._load_text(guid)
        image = self._load_image(guid)
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_seq_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'images': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tags': torch.tensor(tag).long()
        }

    def _load_text(self, guid):
        """
        根据guid返回文本文件
        """
        # guid = str(guid).strip()
        guid = str(guid).split(',')[0]  # 如果 guid 有逗号，取逗号前的部分
        text_file_path = os.path.join(self.image_dir, f"{guid}.txt")
    
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Text file not found: {text_file_path}")

        with open(text_file_path, 'r', encoding='utf-8', errors="ignore") as f:
            text = f.read().strip()

        # 尝试使用文本数据增强
        if self.mode == 'train':
            text = synonym_replacement(text, n=1)

        return text

    def _load_image(self, guid):
        """
        根据guid返回图像文件.jpg
        """
        # guid = str(guid).strip()
        guid = str(guid).split(',')[0]  # 如果 guid 有逗号，取逗号前的部分
        image_path = os.path.join(self.image_dir, f"{guid}.jpg")
    
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 图像数据增强（仅在训练模式下）
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return transform(image)

def synonym_replacement(text, n=1):
    """
    同义词替换
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)
