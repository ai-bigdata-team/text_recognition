from vietocr.tool.config import Cfg
from trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = False
dataset_params = {
    'name':'hw',
    'data_root':'./vietocr/data_line/',
    'train_annotation':'train_line_annotation.txt',
    'valid_annotation':'test_line_annotation.txt'
}

params = {
         'print_every':200,
         'valid_every':15*200,
          'iters':20000,
          'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
          'export':'./weights/transformerocr',
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['dim_feedforward'] = 256*4
config['max_seq_length'] = 256
config['num_encoder_layes'] = 2
config['num_decoder_layers'] = 4
config['device'] = 'cuda:0'

trainer = Trainer(config, pretrained=True)
trainer.config.save('models/vgg_transformer.yml')
trainer.load_weights('weights/transformerocr.pth')
trainer.train()
print(f"model saved successfully")
