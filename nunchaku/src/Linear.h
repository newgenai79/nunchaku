#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"

class GEMV_AWQ : public Module {
public:
    GEMV_AWQ(int in_features, int out_features, bool use_bias, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x);

protected:
    virtual void loadParam(std::string key, Tensor &dst, Tensor src) override;

public:
    const int in_features;
    const int out_features;
    const int group_size;

    int lora_rank;
    float lora_scale;

public:
    Tensor qweight;
    Tensor wscales;
    Tensor wzeros;
    Tensor bias;

    Tensor lora_down;
    Tensor lora_up;

    // std::shared_ptr<CUBLASWrapper> cublas;
};

class GEMM_W4A4 : public Module {
public:
    enum class FuseOptions {
        EMPTY = 0,
        GELU_QUANT,
    };
    struct QuantizedActivation {
        Tensor act;
        Tensor ascales;
        Tensor lora_act;
        bool is_unsigned = false;
    };

public:
    GEMM_W4A4(int in_features, int out_features, bool bias, Tensor::ScalarType dtype, Device device);
    std::variant<Tensor, QuantizedActivation> forward(Tensor x, FuseOptions fuse = FuseOptions::EMPTY, GEMM_W4A4 *nextGEMM = nullptr);
    void forward(Tensor x, Tensor out, Tensor pool = {}, Tensor norm_q = {}, Tensor norm_k = {}, Tensor rotary_emb = {});
    std::variant<Tensor, QuantizedActivation> forward_quant(QuantizedActivation qact, FuseOptions fuse = FuseOptions::EMPTY, GEMM_W4A4 *nextGEMM = nullptr);

public:
    QuantizedActivation quantize(Tensor x);

public:
    const int in_features;
    const int out_features;
    int lora_rank;
    std::vector<float> lora_scales; // every 16 ranks share a scale

    const Tensor::ScalarType dtype;

protected:
    virtual void loadParam(std::string key, Tensor &dst, Tensor src) override;

public:
    Tensor qweight;
    Tensor wscales;
    Tensor bias;

    Tensor lora_down;
    Tensor lora_up;

    Tensor smooth;

    cublasHandle_t handle;
};

// TODO
class GEMM_W8A8;