config = {
    "experiment_name": "car_cd_曲率_20000_8000_7.11",
    
    "training": {
        "batch_size": 32,
        "num_workers": 8,
        "epoch_number": 300,
        "best_val_loss": 0.12,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001
    },
    
    "testing": {
        "batch_size": 32,
        "num_workers": 8,
        "epoch_number": 200,
        "best_test_loss": 0.12,
        "learning_rate": 0.0001,
        "weight_decay": 0.00001
    },
    
    "model": {
        "dim": 384,
        "heads": 6,
        "dim_head": 64,  # 16
        "group_size": 200,
        "num_group": 200
    },
    
    "paths": {
        "stl_path": "/home/j/桌面/agent_design/cd_utils/data/stl_in",  # stl文件路径
        "model_path": "/home/j/桌面/agent_design/cd_utils/best_model_mre_0.0209375933.pt",  # 默认使用与当前网络匹配的权重
        "output": "/home/j/桌面/agent_design/cd_utils/data/output/result.csv"  # 存放测试结果csv的文件夹
    }
}
