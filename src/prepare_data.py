import pickle
import argparse
import yaml


def load_dataset(cfg):
    with open(cfg.data.path, "r") as f:
        dataset = pickle.load(f)
    return dataset


def split_dataset(cfg):
    



def get_args():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--config_path", default="../config/config.yaml", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    


if __name__ == "__main__":
    main()