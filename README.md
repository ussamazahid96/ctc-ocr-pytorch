# ctc-ocr-pytorch
This repo is based on <a href="https://github.com/Xilinx/pytorch-ocr" target="_blank"> pytorch-ocr </a> and traines a Quantized as well as Full Precision BiLSTM with CTCLoss on squential mnist for OCR using pytorch. Visit the original repo for installation and other details. This is to be used with <a href="https://github.com/ussamazahid96/LSTM-PYNQ" target="_blank"> my LSTM-PYNQ fork</a>.

## Data
<img src="sample_image.png" width="224" height="32">

## Quantized Training
Partially Quantized
```
python main.py -p quantized_settings/W4A8/partial.json
```
Fully Quantized Fine Tuning
```
python main.py -p quantized_settings/W4A8/finetune.json --init_bn_fc_fusion --resume <path to .tar>
```
## Evaluate
```
python main.py --resume <path to .tar> --eval
```
## Export Model with PEs and SIMD factor

```
python main.py --resume <path to .tar>  --export --pe 1 --simd_factor 8
```

## Export test image
```
python main.py --resume <path to .tar>  --export_image
```

### Speacial Thanks to
* <a href="https://github.com/stardut/ctc-ocr-tensorflow" target="_blank"> ctc-ocr-tensorflow </a>