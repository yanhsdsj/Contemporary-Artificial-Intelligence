import torch

class Config:
    """
    基础参数设置，包括保存的文件路径
    """
    def __init__(self):
        self.do_train = True
        self.do_test = True

        self.train_file = r'C:\Users\admin\Desktop\project\data\train.txt'
        self.test_file = r'C:\Users\admin\Desktop\project\data\test_without_label.txt'
        self.image_data_dir = r'C:\Users\admin\Desktop\project\data\image_data'
        self.test_output_dir = r'C:\Users\admin\Desktop\project\result'
        self.save_model_path_dir = r'C:\Users\admin\Desktop\project\result'

        self.split_rate = 0.7
        self.pretrained_model = 'bert-base-uncased'
        self.hidden_size = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dropout = 0.3

        self.epochs = 10
        self.batch_size = 32
        self.lr = 1e-5
        self.warmup_steps = 1000
        self.max_seq_length = 512 

        self.img_size = 224
        self.img_channels = 3 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mode = 'train'        
        # self.mode = 'test'