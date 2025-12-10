import os
import pandas as pd
from dataset import get_datalist, load_metadata, load_split

def load_train_val_split(args, preprocessed):
    print("loading data")
    train_dataset, coef_norm = get_datalist(
        root=args.data_dir,
        samples=None, 
        metadata_file=args.metadata_file,
        split_file=args.train_split,
        norm=True,
        coef_norm=None,
        savedir=args.save_dir,
        preprocessed=preprocessed
    )

    val_dataset = get_datalist(
        root=args.data_dir,
        samples=None,
        metadata_file=args.metadata_file,
        split_file=args.val_split,
        norm=False,
        coef_norm=coef_norm,
        savedir=args.save_dir,
        preprocessed=preprocessed
    )

    print("load data finish")
    return train_dataset, val_dataset, coef_norm


def load_test_split(args, preprocessed):
    print("loading data")
    df = pd.read_csv(args.test_split, header=None)
    vallst = df.iloc[:, 0].tolist()  

    test_dataset, coef_norm = get_datalist(
        root=args.data_dir,
        samples=vallst,
        metadata_file=args.metadata_file,
        split_file=args.test_split,
        norm=True,
        coef_norm=None,
        savedir=args.save_dir,
        preprocessed=preprocessed
    )
    print("load data finish")
    return test_dataset, coef_norm, vallst


