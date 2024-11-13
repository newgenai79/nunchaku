#pragma once

#include <torch/extension.h>

#include "common.h"
#include "Tensor.h"

class BufferTorchTensor : public Buffer {
public:
    BufferTorchTensor(at::Tensor tensor) : tensor(std::move(tensor)) {
        this->size = this->tensor.numel() * this->tensor.itemsize();
        this->ptr = this->tensor.data_ptr();
        this->device.type = this->tensor.is_cuda() ? Device::CUDA : Device::CPU;
        this->device.idx = this->tensor.get_device();
    }
private:
    at::Tensor tensor;
};

class TorchOpContext {
public:
    TorchOpContext();
    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&) = delete;
    ~TorchOpContext();
};

Tensor from_torch(at::Tensor input);
at::Tensor to_torch(Tensor input);