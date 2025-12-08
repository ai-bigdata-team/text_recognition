# Run on CLI
# python -m venv .venv 
# pip install vietocr 
# nano +45 .venv/lib/python3.12/site-packages/imgaug/imgaug.py

# Đổi np.sctypes["float"] -> np.float16 , tương tự với 2 dòng dưới
import matplotlib.pyplot as plt
from PIL import Image 

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

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
          'export':'./weights/transformerocr.pth',
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['dim_feedforward'] = 256*4
config['max_seq_length'] = 512
config['num_encoder_layes'] = 2
config['num_decoder_layers'] = 4
config['device'] = 'cuda:0'

trainer = Trainer(config, pretrained=True)
trainer.config.save('models/vgg_transformer.yml')
trainer.load_weights('weights/transformerocr.pth')
val_loss = trainer.validate()
acc_full_seq, acc_per_char, cer, wer = trainer.precision(trainer.metrics)

info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f} - cer: {:.4f} - wer: {:.4f}'.format(
    trainer.iter, 
    val_loss, 
    acc_full_seq, 
    acc_per_char, 
    cer, 
    wer)
print(info)