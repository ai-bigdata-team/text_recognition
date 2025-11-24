from vietocr.tool.config import Cfg
from trainer import Trainer
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="train transformer OCR")
    parser.add_argument("--print-every", type=int, default=200, help="print train loss interval")
    parser.add_argument("--valid-every", type=int, default=2000, help="validation interval")
    parser.add_argument("--max-steps", type=int, default=100000, help="max steps")
    parser.add_argument("--data-root", required=True, help="data folder")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    dataset_params = {
        'name':'hw',
        'data_root': args.data_root,
        'train_annotation':'train_line_annotation.txt',
        'valid_annotation':'test_line_annotation.txt'
    }
    
    params = {
             'print_every':args.print_every,
             'valid_every':args.valid_every,
              'iters':args.max_steps,
              'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
              'export':'models/transformerocr2',
              'metrics': 10000
             }
    
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['dim_feedforward'] = 256*4
    config['max_seq_length'] = 256
    config['num_encoder_layes'] = 4
    config['num_decoder_layers'] = 4
    config['device'] = 'cuda:0'
    
    trainer = Trainer(config, pretrained=True)
    trainer.config.save('models/vgg_transformer.yml')
    trainer.train()
    print(f"model saved successfully")
