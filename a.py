import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
from torch import nn


def main():
    from calflops import calculate_flops
    from torchvision import models

    model = models.alexnet()
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

main()