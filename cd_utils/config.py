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
        "stl_path": "/student/jxy/easy_agent/analy/cd_utils/data/stl_in",  # stl文件路径
        "csv_path": "/student/jxy/easy_agent/analy/cd_utils/data/DrivAer_model_TrainingData_100.csv",  # 带有所有stl文件名和对应cd值的csv路径
        "data_process_path": "/student/jxy/easy_agent/analy/cd_utils/data/car_cur_20000_txt/",  # 处理后的输出txt文件夹
        "csv_process_path": "/student/jxy/easy_agent/analy/cd_utils/data/car_cur_20000_csv/",  # 存放处理后的csv文件夹路径
        "split_path": "/student/jxy/easy_agent/analy/cd_utils/data/DrivAer_model_TrainingData_100.csv",  # 处理后的csv文件路径
        "train_csv_name": "train.csv",  # 训练集csv文件
        "val_csv_name": "val.csv",  # 验证集csv文件
        "test_csv_name": "test.csv",
        "test_csv": "/student/jxy/easy_agent/analy/cd_utils/data/DrivAer_model_TrainingData_100.csv",
        "checkpoint_path": "./checkpoint/",
        "log_dir": "./log_dir/",
        "data_dir": "/student/ysm/data/test_stls/",  # 测试多个stl文件的文件夹
        "data_dir_1stl": "/student/jxy/easy_agent/analy/cd_utils/data/test_1stl/",  # 单个文件文件夹
        "model_path": "/student/jxy/easy_agent/analy/cd_utils/data/checkpoint/car_cd_曲率_20000_8000_6.18/Jun19_01-07-29/best_model_mre_0.0204328585.pt",  # 测试模型
        "output": "/student/jxy/easy_agent/analy/cd_utils/data/output/"  # 存放测试结果csv的文件夹
    }
}
