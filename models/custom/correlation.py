# # import correlation_cuda
# import torch
# from models.custom import correlation_cuda
# from torch.autograd import Function
# from torch.nn.modules.module import Module


# class CorrelationFunction(Function):
#     def __init__(
#         self,
#         pad_size=3,
#         kernel_size=3,
#         max_displacement=20,
#         stride1=1,
#         stride2=2,
#         corr_multiply=1,
#     ):
#         super(CorrelationFunction, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply
#         # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

#     def forward(self, input1, input2):
#         self.save_for_backward(input1, input2)

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#             output = input1.new()

#             correlation_cuda.forward(
#                 input1,
#                 input2,
#                 rbot1,
#                 rbot2,
#                 output,
#                 self.pad_size,
#                 self.kernel_size,
#                 self.max_displacement,
#                 self.stride1,
#                 self.stride2,
#                 self.corr_multiply,
#             )

#         return output

#     def backward(self, grad_output):
#         input1, input2 = self.saved_tensors

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()

#             grad_input1 = input1.new()
#             grad_input2 = input2.new()

#             correlation_cuda.backward(
#                 input1,
#                 input2,
#                 rbot1,
#                 rbot2,
#                 grad_output,
#                 grad_input1,
#                 grad_input2,
#                 self.pad_size,
#                 self.kernel_size,
#                 self.max_displacement,
#                 self.stride1,
#                 self.stride2,
#                 self.corr_multiply,
#             )

#         return grad_input1, grad_input2


# class Correlation(Module):
#     def __init__(
#         self,
#         pad_size=0,
#         kernel_size=0,
#         max_displacement=0,
#         stride1=1,
#         stride2=2,
#         corr_multiply=1,
#     ):
#         super(Correlation, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply

#     def forward(self, input1, input2):
#         result = CorrelationFunction(
#             self.pad_size,
#             self.kernel_size,
#             self.max_displacement,
#             self.stride1,
#             self.stride2,
#             self.corr_multiply,
#         )(input1, input2)

#         return result
import torch

# Import your CUDA correlation module here
from models.custom import correlation_cuda
from torch.autograd import Function
from torch.nn.modules.module import Module

# TODO : change input shape


class CorrelationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input1,
        input2,
        pad_size,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        corr_multiply,
    ):
        ctx.save_for_backward(input1, input2)
        ctx.other_args = (
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            corr_multiply,
        )

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(
                input1, input2, rbot1, rbot2, output, *ctx.other_args
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        other_args = ctx.other_args

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(
                input1,
                input2,
                rbot1,
                rbot2,
                grad_output,
                grad_input1,
                grad_input2,
                *other_args
            )

        return grad_input1, grad_input2, None, None, None, None, None


class Correlation(Module):
    def __init__(
        self,
        pad_size=0,
        kernel_size=0,
        max_displacement=0,
        stride1=1,
        stride2=2,
        corr_multiply=1,
    ):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return CorrelationFunction.apply(
            input1,
            input2,
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply,
        )
