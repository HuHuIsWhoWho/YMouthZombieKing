# tools/generate_hsi_configs.py
import os
import yaml

def generate_dataset_configs():
    """为每个高光谱数据集生成对应的配置文件"""
    
    datasets = [
        # 数据集名称, 类别数, 波段数, 插值后伪图像大小
        ('Indian_Pines', 16, 200, 64),
        ('Pavia_University', 9, 103, 48),
        ('Pavia_Centre', 9, 102, 48),
        ('Salinas', 16, 204, 64),
        ('KSC', 13, 176, 56),
        ('Botswana', 14, 145, 56),
        ('Houston', 15, 144, 56),
        ('WHU-Hi-HanChuan', 16, 274, 72),
        ('WHU-Hi-HongHu', 22, 270, 72),
        ('WHU-Hi-LongKou', 9, 270, 72),
        ('Trento', 6, 63, 36),
    ]
    
    base_config = {
        'MODEL': {
            'TYPE': 'vHeat4HSI',
            'VHEAT': {
                'MLP_RATIO': 2.0,
                'DEPTHS': [1, 2, 2, 1],
            }
        },
        'DATA': {
            'DATASET': 'hsi',
            'BATCH_SIZE': 256,
            'PATCH_SIZE':11,
            'TRAIN_RATIO': 0.15,
            'VAL_RATIO': 0.1,
        },
        'TRAIN': {
            'EPOCHS': 600,
            'BASE_LR': 1e-3,  # 5e-4
            'WEIGHT_DECAY': 0.05,  
        },
        'SAVE_FREQ': 10,
        'PRINT_FREQ': 10,
        'OUTPUT': './output_hsi',
        'SEED': 42,
    }
    
    for dataset_name, num_classes, bands, img_size in datasets:
        config = base_config.copy()
        
        # 更新数据集特定配置
        config['MODEL']['NAME'] = f'vHeat4HSI_{dataset_name}'
        config['MODEL']['NUM_CLASSES'] = num_classes
        config['MODEL']['VHEAT']['EMBED_DIM'] = int(bands * 1.0)  # 嵌入维度设为波段数的100%
        
        config['DATA']['HSI_DATASET'] = dataset_name
        config['DATA']['DATA_PATH'] = './dataset/HSI_data'
        config['DATA']['BANDS'] = bands

        config['OUTPUT'] = f'./output_hsi/{dataset_name}'
        
        # 保存配置文件
        config_file = f'configs/vHeat4HSI/vheat4hsi_{dataset_name.lower()}.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        

if __name__ == '__main__':
    generate_dataset_configs()