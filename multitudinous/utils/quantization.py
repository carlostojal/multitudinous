import torch
from torch import nn

def get_quantized_model(model_fp32: nn.Module, backend: str = 'x86') -> nn.Module:

    if backend not in ['x86', 'qnnpack']:
        raise ValueError(f'Quantization backend {backend} not recognized. Available backends are \'x86\' and \'qnnpack\'.')

    # set the quantization backend
    torch.backends.quantized.engine = backend
    model_fp32.qconfig = torch.quantization.get_default_qconfig(backend)

    # prepare the model for quantization
    model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace=False)

    # convert the model to a quantized model
    model_fp32_quantized = torch.quantization.convert(model_fp32_prepared, inplace=False)

    del model_fp32_prepared

    return model_fp32_quantized