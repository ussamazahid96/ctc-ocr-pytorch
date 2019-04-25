# ctc-ocr-pytorch
This repo traines a BiLSTM with CTCLoss on squential mnist for OCR using pytorch

## Requirements

 * python 3.5+
 * numpy
 * pytorch (version >= 1.0)
 * torchvision
 * <a href="https://github.com/SeanNaren/warp-ctc" target="_blank"> warpctc_pytorch </a>

## Data
<img src="sample_image.png" width="224" height="28">

## Training

```
python main.py 
```
Or resume from a checkpoint
```
python main.py --resume
```

## Evaluate
```
python main.py --eval
```

### Speacial Thanks to
* <a href="https://github.com/Xilinx/pytorch-ocr" target="_blank"> pytorch-ocr </a>
* <a href="https://github.com/stardut/ctc-ocr-tensorflow" target="_blank"> ctc-ocr-tensorflow </a>
