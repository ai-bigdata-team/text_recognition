import os
from config import Config
from models import Trainer
import argparse

'''
python train.py \
    --dataset-name mcocr \
    --valid-every 300 \
    --max-steps 7000 \
    --data-root datasets/MCOCR/high_contrast \
    --logging-dir training_mcocr
'''

def get_args():
    parser = argparse.ArgumentParser(description="train transformer OCR")
    parser.add_argument("--dataset-name", default = "unknown", help="Name of the dataset")
    parser.add_argument("--print-every", type=int, default=200, help="print train loss interval")
    parser.add_argument("--valid-every", type=int, default=2000, help="validation interval")
    parser.add_argument("--max-steps", type=int, default=100000, help="max steps")
    parser.add_argument("--data-root", required=True, help="data folder")
    parser.add_argument("--dataset-prefix", default="text_recognition", help="prefix for training and validation data")
    parser.add_argument("--logging-dir", default="training_info", help="Directory for logging")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    config = Config.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    dataset_params = {
        'name': args.dataset_name,
        'data_root': args.data_root,
        'train_annotation':f'{args.dataset_prefix}_train.txt',
        'valid_annotation':f'{args.dataset_prefix}_val.txt'
    }
    
    params = {
             'print_every':args.print_every,
             'valid_every':args.valid_every,
              'iters':args.max_steps,
              'checkpoint':f'logs/{args.logging_dir}/checkpoints',    
              'export':f'logs/{args.logging_dir}/checkpoints/transformerocr',
              'logging_dir': f'logs/{args.logging_dir}',
              'metrics': 1000
             }
    
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['dim_feedforward'] = 256*4
    config['max_seq_length'] = 256
    config['num_encoder_layes'] = 4
    config['num_decoder_layers'] = 4
    config['device'] = 'cuda:0'
    
    trainer = Trainer(config, pretrained=True)
    os.makedirs(f'logs/{args.logging_dir}/config', exist_ok=True)
    trainer.config.save(f'logs/{args.logging_dir}/config/model_config.yml')
    trainer.train()
    print(f"model saved successfully")
