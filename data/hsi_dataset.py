import os
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class HSIDataset(Dataset):
    """
    高光谱图像分类数据集
    支持多种标准高光谱数据集格式
    """
    
    def __init__(self, data_root, dataset_name, mode='train', transform=None, 
                 train_ratio=0.2, val_ratio=0.1, seed=42, patch_size=7):
        """
        Args:
            data_root: 数据根目录
            dataset_name: 数据集名称 (如 'Indian_Pines', 'PaviaU', 'Salinas', 等)
            mode: 'train', 'val', 'test'
            transform: 数据增强变换
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        super().__init__()
        
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.patch_size = patch_size
        
        # 加载数据
        self.data, self.labels, self.num_classes = self._load_dataset()

        # 选择波段（如果需要）
        # 这里假设已经有预先选择的波段索引文件 'fisher_selected_bands.npz'
        # band_indices = np.load('fisher_selected_bands.npz')['selected_indices']
        # self.data = self.data[:, :, band_indices]

        # 图片内归一化
        # self.data = (self.data - np.mean(self.data)) / (np.std(self.data) + 1e-8)

        # 数据集划分
        self.indices = self._split_dataset(train_ratio, val_ratio, seed)
        
        print(f"{dataset_name} - {mode}: {len(self.indices)} samples, {self.num_classes} classes")
    
    def _load_dataset(self):
        """根据数据集名称加载对应的数据"""
        dataset_path = os.path.join(self.data_root, self.dataset_name)
        
        if self.dataset_name == 'Indian_Pines':
            return self._load_indian_pines(dataset_path)
        elif self.dataset_name == 'Pavia_University':
            return self._load_pavia_u(dataset_path)
        elif self.dataset_name == 'Pavia_Centre':
            return self._load_pavia_centre(dataset_path)
        elif self.dataset_name == 'Salinas':
            return self._load_salinas(dataset_path)
        elif self.dataset_name == 'KSC':
            return self._load_ksc(dataset_path)
        elif self.dataset_name == 'Botswana':
            return self._load_botswana(dataset_path)
        elif self.dataset_name == 'Houston':
            return self._load_houston(dataset_path)
        elif 'WHU-Hi' in self.dataset_name:
            return self._load_whu_hi(dataset_path, self.dataset_name)
        elif self.dataset_name == 'Trento':
            return self._load_trento(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_indian_pines(self, path):
        """加载Indian Pines数据集"""
        try:
            data = sio.loadmat(os.path.join(path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
            labels = sio.loadmat(os.path.join(path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        except:
            data = sio.loadmat(os.path.join(path, 'Indian_pines.mat'))['indian_pines']
            labels = sio.loadmat(os.path.join(path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        
        return data, labels, np.max(labels)
            
    def _load_pavia_u(self, path):
        """加载Pavia University数据集"""
        data = sio.loadmat(os.path.join(path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(path, 'PaviaU_gt.mat'))['paviaU_gt']
        return data, labels, np.max(labels)
    
    def _load_pavia_centre(self, path):
        """加载Pavia Centre数据集"""
        data = sio.loadmat(os.path.join(path, 'Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(path, 'Pavia_gt.mat'))['pavia_gt']
        return data, labels, np.max(labels)
    
    def _load_salinas(self, path):
        """加载Salinas数据集"""
        data = sio.loadmat(os.path.join(path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(path, 'Salinas_gt.mat'))['salinas_gt']
        return data, labels, np.max(labels)
    
    def _load_ksc(self, path):
        """加载KSC数据集"""
        data = sio.loadmat(os.path.join(path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(path, 'KSC_gt.mat'))['KSC_gt']
        return data, labels, np.max(labels)
    
    def _load_botswana(self, path):
        """加载Botswana数据集"""
        data = sio.loadmat(os.path.join(path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(path, 'Botswana_gt.mat'))['Botswana_gt']
        return data, labels, np.max(labels)
    
    def _load_houston(self, path):
        """加载Houston数据集"""
        data = sio.loadmat(os.path.join(path, 'Houstondata.mat'))['houstondata']
        labels = sio.loadmat(os.path.join(path, 'Houstonlabel.mat'))['houstonlabel']
        return data, labels, np.max(labels)
    
    def _load_whu_hi(self, path, dataset_name):
        """加载WHU-Hi系列数据集"""
        if 'HanChuan' in dataset_name:
            data = sio.loadmat(os.path.join(path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
            labels = sio.loadmat(os.path.join(path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
        elif 'HongHu' in dataset_name:
            data = sio.loadmat(os.path.join(path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
            labels = sio.loadmat(os.path.join(path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
        elif 'LongKou' in dataset_name:
            data = sio.loadmat(os.path.join(path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
            labels = sio.loadmat(os.path.join(path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        else:
            raise ValueError(f"Unknown WHU-Hi dataset: {dataset_name}")
        
        return data, labels, np.max(labels)
    
    def _load_trento(self, path):
        """加载Trento数据集"""
        data = sio.loadmat(os.path.join(path, 'Italy_hsi.mat'))['Italy_hsi']
        labels = sio.loadmat(os.path.join(path, 'allgrd.mat'))['allgrd']
        return data, labels, np.max(labels)
    
    def _split_dataset(self, train_ratio, val_ratio, seed):
        """划分训练集、验证集、测试集"""
        np.random.seed(seed)
        
        # 获取所有有效像素位置（标签不为0）
        valid_positions = np.argwhere(self.labels > 0)
        
        # 打乱顺序
        np.random.shuffle(valid_positions)
        
        total_samples = len(valid_positions)
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)
        
        if self.mode == 'train':
            return valid_positions[:train_samples]
        elif self.mode == 'val':
            return valid_positions[train_samples:train_samples + val_samples]
        else:  # test
            return valid_positions[train_samples + val_samples:]
    
    def _extract_patch(self, h, w, patch_size=7):
        """提取以(h,w)为中心的patch"""
        half = patch_size // 2
        
        # 处理边界情况
        h_start = max(0, h - half)
        h_end = min(self.data.shape[0], h + half + 1)
        w_start = max(0, w - half)
        w_end = min(self.data.shape[1], w + half + 1)
        
        # 提取patch
        patch = self.data[h_start:h_end, w_start:w_end, :]
        
        # 如果patch大小不够，进行padding
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            new_patch = np.zeros((patch_size, patch_size, self.data.shape[2]), 
                                dtype=self.data.dtype)
            pad_h = (patch_size - patch.shape[0]) // 2
            pad_w = (patch_size - patch.shape[1]) // 2
            new_patch[pad_h:pad_h+patch.shape[0], 
                     pad_w:pad_w+patch.shape[1], :] = patch
            patch = new_patch
        
        # 转换为CHW格式
        patch = np.transpose(patch, (2, 0, 1))

        # 转为 float32，按通道 standardize（更稳健）
        patch = patch.astype(np.float32)
        
        return patch
    
    def __getitem__(self, idx):
        h, w = self.indices[idx]
        label = self.labels[h, w] - 1  # 标签从0开始
        
        # 提取patch（可以根据需要调整patch大小）
        patch = self._extract_patch(h, w, patch_size=self.patch_size)
        
        # 转为 tensor（使用 from_numpy 保留内存共享），并确保 contiguous
        patch = torch.from_numpy(patch).float().contiguous()
        label = torch.tensor(label, dtype=torch.long)
        
        return patch, label
    
    def __len__(self):
        return len(self.indices)


class HSIDataManager:
    """
    高光谱数据管理器，统一管理多个数据集
    """
    
    DATASET_INFO = {
        'Indian_Pines': {
            'bands': 200,
            'height': 145,
            'width': 145,
            'classes': 16,
            'patch_size': 11
        },
        'Pavia_University': {
            'bands': 103,
            'height': 610,
            'width': 340,
            'classes': 9,
            'patch_size': 7
        },
        'Pavia_Centre': {
            'bands': 102,
            'height': 1096,
            'width': 715,
            'classes': 9,
            'patch_size': 7
        },
        'Salinas': {
            'bands': 204,
            'height': 512,
            'width': 217,
            'classes': 16,
            'patch_size': 7
        },
        'KSC': {
            'bands': 176,
            'height': 512,
            'width': 614,
            'classes': 13,
            'patch_size': 7
        },
        'Botswana': {
            'bands': 145,
            'height': 1476,
            'width': 256,
            'classes': 14,
            'patch_size': 7
        },
        'Houston': {
            'bands': 144,
            'height': 349,
            'width': 1905,
            'classes': 15,
            'patch_size': 7
        },
        'WHU-Hi-HanChuan': {
            'bands': 274,
            'height': 1217,
            'width': 303,
            'classes': 16,
            'patch_size': 7
        },
        'WHU-Hi-HongHu': {
            'bands': 270,
            'height': 1217,
            'width': 303,
            'classes': 22,
            'patch_size': 7
        },
        'WHU-Hi-LongKou': {
            'bands': 270,
            'height': 550,
            'width': 400,
            'classes': 9,
            'patch_size': 7
        },
        'Trento': {
            'bands': 63,
            'height': 166,
            'width': 600,
            'classes': 6,
            'patch_size': 7
        }
    }
    
    @classmethod
    def get_dataset_info(cls, dataset_name):
        """获取数据集信息"""
        return cls.DATASET_INFO.get(dataset_name, {
            'bands': 200,
            'height': 100,
            'width': 100,
            'classes': 10,
            'patch_size': 7
        })
    
    @classmethod
    def list_datasets(cls):
        """列出所有支持的数据集"""
        return list(cls.DATASET_INFO.keys())
    