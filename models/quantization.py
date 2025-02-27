import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

HARDENING_RATIO = 0.05
N_BITS = 8

class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

    def forward(self, input):
        if self.inf_with_weight:
            return F.conv2d(input, self.weight * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.conv2d(input, weight_quan, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

class quan_HardenedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, hardening_ratio=HARDENING_RATIO, N_bits=N_BITS):
        super(quan_HardenedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  dilation=dilation, groups=groups, bias=bias)
        
        # Quantization parameters
        self.N_bits = N_bits
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.inf_with_weight = False  # Disable by default
        self.__reset_stepsize__()

        # Bit-weight vector for bit-flipping
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)
        self.b_w[0] = -self.b_w[0]  # In-place change MSB to negative
        
        # Hardening parameters
        self.hardening_ratio = hardening_ratio
        self.duplicated_channels = int(out_channels * hardening_ratio)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #self.device = torch.device("cpu")
        
        # Ensure duplication count is at least 1 if hardening is applied
        if self.duplicated_channels > 0:
            self.__duplicate_channels__()
    
    def __duplicate_channels__(self):
        """ Duplicate channels for fault tolerance """
        duplication_count = self.duplicated_channels
        new_out_channels = self.out_channels + duplication_count
        
        _, b, c, d = self.weight.size()
        new_weight = torch.zeros(new_out_channels, b, c, d, device=self.device)
        new_bias = torch.zeros(new_out_channels, device=self.device) if self.bias is not None else None
        
        # Duplicate the first `duplication_count` channels
        with torch.no_grad():
            for i in range(duplication_count):
                new_weight[i] = self.weight[i]
                new_weight[duplication_count + i] = self.weight[i]
                if new_bias is not None:
                    new_bias[i] = self.bias[i]
                    new_bias[duplication_count + i] = self.bias[i]
            new_weight[2 * duplication_count:] = self.weight[duplication_count:]
            if new_bias is not None:
                new_bias[2 * duplication_count:] = self.bias[duplication_count:]
        
        self.weight = nn.Parameter(new_weight)
        if new_bias is not None:
            self.bias = nn.Parameter(new_bias)
        
        self.out_channels = new_out_channels
    
    def forward(self, input):
        if self.inf_with_weight:
            out_activation = F.conv2d(input, self.weight * self.step_size, self.bias,
                                      self.stride, self.padding, self.dilation, self.groups)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size, self.half_lvls) * self.step_size
            out_activation = F.conv2d(input, weight_quan, self.bias, self.stride,
                                      self.padding, self.dilation, self.groups)
        
        # If no duplication, return as is
        if self.duplicated_channels == 0:
            return out_activation
        
        # Apply correction mechanism for duplicated channels
        correction_mask = torch.ge(out_activation[:, :self.duplicated_channels, :, :],
                                   out_activation[:, self.duplicated_channels:2 * self.duplicated_channels, :, :])
        corrected_results = out_activation[:, :self.duplicated_channels] * torch.logical_not(correction_mask) + \
                            out_activation[:, self.duplicated_channels:2 * self.duplicated_channels] * correction_mask
        
        # Remove redundant channels after correction
        batch, out_ch, w, h = out_activation.size()
        new_out_activation = torch.zeros((batch, out_ch - self.duplicated_channels, w, h), device=self.device)
        new_out_activation[:, :self.duplicated_channels] = corrected_results
        new_out_activation[:, self.duplicated_channels:] = out_activation[:, 2 * self.duplicated_channels:]
        
        return new_out_activation
    
    def __reset_stepsize__(self):
        """ Reset the quantization step size based on the weight range """
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls
    
    def __reset_weight__(self):
        """ Convert the weight tensor to its quantized form permanently """
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
        self.inf_with_weight = True

class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True
