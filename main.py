import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='CPSP/config/avel.yaml')
    parser.add_argument('--baseline', type=str, choices=['AVEL', 'CPSP', 'CMBS', 'LAVISH'], default='AVEL', help='AVEL, CPSP, CMBS or LAVISH')
    
    # MARK: hyperparameters
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    # MARK: train
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size')


    # MARK: path



    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    args = parse_args()
    # cfg = load_config(args.config)
    # override parameters from command line
    selected_baseline = args.baseline
    if selected_baseline == 'AVEL':
        pass
    elif selected_baseline == 'CPSP':
        pass
    elif selected_baseline == 'CMBS':

        pass
    elif selected_baseline == 'LAVISH':
        pass
    

    # 覆盖 yaml 配置中的参数
    

    # print(cfg)
