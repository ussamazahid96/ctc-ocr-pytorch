import os
import json
import torch
import argparse
from trainer import Seq_MNIST_Trainer

torch.backends.cudnn.enabled = False
torch.set_printoptions(precision=10)

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) if isinstance(pair[1], unicode) else pair for pair in data.items())


def non_or_str(value):
    if value == None:
        return None
    return value

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Quantized BiLSTM Sequential MNIST Example')
    parser.add_argument('--params', '-p', type=str, default="default_trainer_params.json", help='Path to params JSON file. Default ignored when resuming.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--init_bn_fc_fusion', default=False, action='store_true', help='Init BN FC fusion.')
    parser.add_argument('--resume', type=non_or_str, help='resume from a checkpoint')
    parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
    parser.add_argument('--export', default=False, action='store_true', help='perform weights export as .hpp of trained model')
    parser.add_argument('--export_image', default=False, action='store_true', help='perform test image export as png and txt')
    parser.add_argument('--experiments', default="./experiments", help='Save Path')
    parser.add_argument('--simd_factor', default=1, type=int, help='SIMD factor for export.')
    parser.add_argument('--pe', default=1, type=int, help='Number of PEs for export.')
    
    #Overrides
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--num_units', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--word_size', type=int)
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--neuron_type', type=str)
    parser.add_argument('--input_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--bidirectional', type=bool)
    parser.add_argument('--reduce_bidirectional', type=str)
    parser.add_argument('--recurrent_bias_enabled', type=bool)
    parser.add_argument('--checkpoint_interval', type=int)
    parser.add_argument('--recurrent_weight_bit_width', type=int)
    parser.add_argument('--recurrent_weight_quantization', type=str)
    parser.add_argument('--recurrent_bias_bit_width', type=int)
    parser.add_argument('--recurrent_bias_quantization', type=str)
    parser.add_argument('--recurrent_activation_bit_width', type=int)
    parser.add_argument('--recurrent_activation_quantization', type=str)
    parser.add_argument('--internal_activation_bit_width', type=int)
    parser.add_argument('--fc_weight_bit_width', type=int)
    parser.add_argument('--fc_weight_quantization', type=str)
    parser.add_argument('--fc_bias_bit_width', type=int)
    parser.add_argument('--fc_bias_quantization', type=str)
    parser.add_argument('--quantize_input', type=bool)

    args = parser.parse_args()
    if args.export:
        args.no_cuda = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.experiments):
        os.mkdir(args.experiments)

    if (args.resume or args.eval or args.export) and args.params == "default_trainer_params.json":
        package = torch.load(args.resume, map_location=lambda storage, loc: storage)
        trainer_params = package['trainer_params']
    else:
        with open(args.params) as d:
            trainer_params = json.load(d, object_hook=ascii_encode_dict)
    trainer_params = objdict(trainer_params)

    for k in trainer_params.keys():
        print(k, trainer_params[k])

    trainer = Seq_MNIST_Trainer(trainer_params, args)

    if args.export:
        trainer.export_model(args.simd_factor, args.pe)
        exit(0)

    if args.export_image:
        trainer.export_image()
        exit(0)

    if args.eval:
        trainer.eval_model()
        exit(0)
 
    else:
        trainer.train_model()
