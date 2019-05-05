# ctc-ocr-pytorch
This repo is based on <a href="https://github.com/Xilinx/pytorch-ocr" target="_blank"> pytorch-ocr </a> and traines a Quantized as well as Full Precision BiLSTM with CTCLoss on squential mnist for OCR using pytorch. Visit the original repo for installation and other details. This is to be used with <a href="https://github.com/ussamazahid96/LSTM-PYNQ" target="_blank"> my LSTM-PYNQ fork</a>.

## Data
<img src="sample_image.jpg" width="224" height="28">

## Quantized Training
Partially Quantized
```
python main.py -p q_params/p_W4A4.json
```
Fully Quantized Fine Tuning
```
python main.py -p q_params/f_W4A4.json --init_bn_fc_fusion --resume
```
## Evaluate
```
python main.py --eval
```
## Export Model

```
python main.py --export
```

## Export test image
```
python main.py --export_image
```

### Speacial Thanks to
* <a href="https://github.com/stardut/ctc-ocr-tensorflow" target="_blank"> ctc-ocr-tensorflow </a>