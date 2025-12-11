# TEXT RECOGNITION
Developing a deep learning model for recognizing text from image data.

## Installation
`Python 3.12.3`
```
python -m venv .venv 
pip install -r requirements.txt
``` 

Notes: vietocr using old version, so, to run code, you have to modify source code in packages: 
```
nano +45 .venv/lib/python3.12/site-packages/imgaug/imgaug.py
```
Change np.sctypes["float"] -> np.float16 , Do the same with two following lines

## Training 
```
python train.py \
    --dataset-name mcocr \
    --valid-every 300 \
    --max-steps 7000 \
    --data-root datasets/MCOCR/high_contrast \
    --logging-dir training_mcocr
```

## Evaluation
```
python evaluate.py \
    --dataset-name mcocr \
    --valid-every 300 \
    --max-steps 7000 \
    --data-root datasets/MCOCR/high_contrast \
    --logging-dir training_mcocr
```