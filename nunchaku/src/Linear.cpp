#include "Linear.h"
#include "kernels/gemm_w4a4.h"
#include "kernels/gemm_f16.h"
#include "kernels/misc_kernels.h"
#include "kernels/awq/gemv_awq.h"

GEMV_AWQ::GEMV_AWQ(int in_features, int out_features, bool use_bias, Tensor::ScalarType dtype, Device device) : 
    in_features(in_features), out_features(out_features), group_size(64), lora_rank(0), lora_scale(1.0f)
{
    this->qweight = Tensor::allocate({out_features / 4, ceilDiv(in_features, 8) * 4}, Tensor::INT32, device);
    this->wscales = Tensor::allocate({ceilDiv(in_features, group_size), out_features}, dtype, device);
    this->wzeros  = Tensor::allocate({ceilDiv(in_features, group_size), out_features}, dtype, device);
    this->bias = use_bias ? Tensor::allocate({out_features}, dtype, device) : Tensor{};

    // !!! lora layout is different from w4a4 !!!
    this->lora_down = Tensor::allocate({lora_rank, in_features}, dtype, device, true);
    this->lora_up = Tensor::allocate({out_features, lora_rank}, dtype, device, true);

    registerParams
        (qweight, "qweight")
        (wscales, "wscales")
        (wzeros, "wzeros")
        (bias, "bias")
        (lora_down, "lora_down", ParamFlags::Optional)
        (lora_up, "lora_up", ParamFlags::Optional)
    ;
}

void GEMV_AWQ::loadParam(std::string key, Tensor &dst, Tensor src) {
    if (key == "lora_down" || key == "lora_up") {
        assert(src.ndims() == 2);
        if (dst.shape.dataExtent != src.shape.dataExtent) {
            dst = src.copy(this->qweight.device());
            if (key == "lora_down") {
                const int new_rank = dst.shape[0];
                this->lora_rank = new_rank;
            }
        } else {
            dst.copy_(src);
        }
    } else {
        Module::loadParam(key, dst, src);
    }
}

Tensor GEMV_AWQ::forward(Tensor x) {
    debug("x", x);

    const int M = (int)x.numel() / x.shape[-1];
    Tensor out = gemv_awq(x, this->qweight, this->wscales, this->wzeros, M, out_features, in_features, group_size);
    if (bias.valid()) {
        // TODO: batch
        assert(out.numel() == bias.numel());
        out = add(out, bias.view(out.shape.dataExtent));
    }

    debug("out_before_lora", out);

    if (this->lora_rank > 0) {
        Tensor lora_act = gemm_f16(x, this->lora_down, {}, 1.0f, 0.0f);
        debug("lora_act", lora_act);

        Tensor lora_out = gemm_f16(lora_act, this->lora_up, {}, this->lora_scale, 0.0f);
        debug("lora_out", lora_out);

        out = add(out, lora_out);
    }

    debug("out", out);
    
    return out;
}


#define NO_LORA_FUSION 0

GEMM_W4A4::GEMM_W4A4(int in_features, int out_features, bool bias, Tensor::ScalarType dtype, Device device) : 
    in_features(in_features), out_features(out_features), lora_rank(0), dtype(dtype)
{
    this->qweight = Tensor::allocate({out_features, in_features / 2}, Tensor::INT8, device, true);
    this->wscales = Tensor::allocate({in_features / 64, out_features}, dtype, device, true);

    this->bias = bias ? Tensor::allocate({out_features}, dtype, device, true) : Tensor{};

    this->lora_down = Tensor::allocate({in_features, lora_rank}, dtype, device, true);
    this->lora_up = Tensor::allocate({out_features, lora_rank}, dtype, device, true);

    // TODO: smooth factor in FC1+FC2 fusion
    // TODO: smooth factor in non-Lora fusion
    this->smooth = Tensor::allocate({in_features}, dtype, device, true);

    registerParams
        (qweight, "qweight")
        (wscales, "wscales")
        (this->bias, "bias")
        (lora_down, "lora_down", ParamFlags::Optional)
        (lora_up, "lora_up", ParamFlags::Optional)
        (smooth, "smooth")
    ;

#if NO_LORA_FUSION
    checkCUBLAS(cublasCreate(&handle));
#endif
}

void GEMM_W4A4::loadParam(std::string key, Tensor &dst, Tensor src) {
    if (key == "lora_down" || key == "lora_up") {
        assert(src.ndims() == 2);
        if (dst.shape.dataExtent != src.shape.dataExtent) {
            dst = src.copy(this->qweight.device());
            this->lora_rank = dst.shape[1];
            this->lora_scales.resize(ceilDiv(this->lora_rank, 16), 1.0f);
        } else {
            dst.copy_(src);
        }
    } else {
        Module::loadParam(key, dst, src);
    }
}

std::variant<Tensor, GEMM_W4A4::QuantizedActivation> GEMM_W4A4::forward(Tensor x, FuseOptions fuse, GEMM_W4A4 *nextGEMM) {
    return forward_quant(quantize(x), fuse, nextGEMM);
}

void GEMM_W4A4::forward(Tensor x, Tensor out, Tensor pool, Tensor norm_q, Tensor norm_k, Tensor rotary_emb) {
    QuantizedActivation qact = quantize(x);

#if !NO_LORA_FUSION

#if 0
    Tensor dummy = Tensor::empty_like(qact.lora_act);
    dummy.zero_();

    gemm_w4a4(qact.act, qweight, out, {}, qact.ascales, wscales, {}, pool, dummy, this->lora_up, {}, {}, norm_q, norm_k, rotary_emb, this->bias, {}, qact.is_unsigned);
    debug("gemm.nolora.out", out);
#endif

    gemm_w4a4(qact.act, qweight, out, {}, qact.ascales, wscales, {}, pool, qact.lora_act, this->lora_up, {}, {}, norm_q, norm_k, rotary_emb, this->bias, {}, qact.is_unsigned, this->lora_scales);

    debug("gemm.out", out);
#else
    const int M = (int)qact.act.numel() / qact.act.shape[-1];

    gemm_w4a4(qact.act, qweight, out, {}, qact.ascales, wscales, {}, pool, {}, {}, {}, {});

    nvtxRangePushA("LoraUp");

    static const half one = 1.0;
    static const half zero = 0.0;
    // lora_up: [M, R] * [OC, R] => [M, OC]
    // cublas view: [OC, R] * [M, R]^T
    checkCUBLAS(cublasHgemm(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_N, 
        this->out_features, M, this->lora_rank,
        &one,
        this->lora_up.data_ptr<half>(),
        this->lora_rank,
        qact.lora_act.data_ptr<half>(),
        this->lora_rank,
        &one, 
        out.data_ptr<half>(),
        this->out_features));

    nvtxRangePop();
#endif
}

std::variant<Tensor, GEMM_W4A4::QuantizedActivation> GEMM_W4A4::forward_quant(QuantizedActivation qact, FuseOptions fuse, GEMM_W4A4 *nextGEMM) {
    Tensor out;
    QuantizedActivation qout;

    Tensor next_lora;
    Tensor next_smooth;

    const int M = (int)qact.act.numel() / qact.act.shape[-1];

    if (fuse == FuseOptions::EMPTY) {
        auto shape = TensorShape(qact.act.shape.dataExtent);
        shape[-1] = out_features;
        out = Tensor::allocate(shape, dtype, qweight.device());
    } else {
        auto shape = TensorShape(qact.act.shape.dataExtent);
        shape[-1] = out_features / 2;
        qout.act = Tensor::allocate(shape, Tensor::INT8, qweight.device());
        qout.ascales = Tensor::allocate({out_features / 64, M}, dtype, qweight.device());
        qout.lora_act = Tensor::allocate({M, lora_rank}, Tensor::FP32, qweight.device());
        qout.is_unsigned = true;

        next_lora = nextGEMM->lora_down;
        next_smooth = nextGEMM->smooth;
    }

#if !NO_LORA_FUSION

#if 0
    Tensor dummy = Tensor::empty_like(qact.lora_act);
    dummy.zero_();

    gemm_w4a4(qact.act, qweight, out, qout.act, qact.ascales, wscales, qout.ascales, {}, dummy, this->lora_up, next_lora, qout.lora_act, {}, {}, {}, this->bias, next_smooth, qact.is_unsigned);

    if (fuse == FuseOptions::EMPTY) {
        debug("gemm.nolora.out", out);
    } else {
        debug("gemm.nolora.qout", qout.act);
        debug("gemm.nolora.oscales", qout.ascales);
        debug("gemm.nolora.lora_act_out", qout.lora_act);
    }
#endif

    gemm_w4a4(qact.act, qweight, out, qout.act, qact.ascales, wscales, qout.ascales, {}, qact.lora_act, this->lora_up, next_lora, qout.lora_act, {}, {}, {}, this->bias, next_smooth, qact.is_unsigned, this->lora_scales);

    if (fuse == FuseOptions::EMPTY) {
        debug("gemm.out", out);
    } else {
        debug("gemm.qout", qout.act);
        debug("gemm.oscales", qout.ascales);
        debug("gemm.lora_act_out", qout.lora_act);
    }

    
#else
    if (!out.valid()) {
        auto shape = TensorShape(qact.act.shape.dataExtent);
        shape[-1] = out_features;
        out = Tensor::allocate(shape, Tensor::FP16, qweight.device());
    }

    gemm_w4a4(qact.act, qweight, out, qout.act, qact.ascales, wscales, qout.ascales, {}, {}, {}, {}, {});

    nvtxRangePushA("LoraUp");

    static const half one = 1.0;
    static const half zero = 0.0;

    // lora_up: [M, R] * [OC, R]^T => [M, OC]
    // cublas view: [R, OC]^T * [R, M] => [OC, M]
    // lora_up layout wrong?
    checkCUBLAS(cublasHgemm(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_N, 
        this->out_features, M, this->lora_rank,
        &one,
        this->lora_up.data_ptr<half>(),
        this->lora_rank,
        qact.lora_act.data_ptr<half>(),
        this->lora_rank,
        &one, 
        out.data_ptr<half>(),
        this->out_features));

    nvtxRangePop();

    if (fuse == FuseOptions::GELU_QUANT) {
        nvtxRangePushA("LoraDown");
        // IC is for next lora (OC of this layer)
        // lora_down: [M, IC] * [IC, R] => [M, R]
        // cublas view: [R, IC] * [IC, M] => [R, M]
        checkCUBLAS(cublasHgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            this->lora_rank, M, this->out_features,
            &one,
            next_lora.data_ptr<half>(),
            this->lora_rank,
            out.data_ptr<half>(),
            this->out_features,
            &zero, 
            qout.lora_act.data_ptr<half>(),
            this->lora_rank));

        out = {};

        nvtxRangePop();
    }

#endif
    if (out.valid()) {
        return out;
    }
    return qout;
}

GEMM_W4A4::QuantizedActivation GEMM_W4A4::quantize(Tensor x) {
    const int M = x.numel() / x.shape[-1];

    auto shape = TensorShape(x.shape.dataExtent);
    shape[-1] = in_features / 2;

    QuantizedActivation qact;
    qact.act = Tensor::allocate(shape, Tensor::INT8, qweight.device());
    qact.ascales = Tensor::allocate({in_features / 64, M}, dtype, qweight.device());
    qact.lora_act = Tensor::allocate({M, lora_rank}, Tensor::FP32, qweight.device());
    qact.is_unsigned = false;

#if !NO_LORA_FUSION
    debug("quantize.x", x);
    debug("quantize.smooth", this->smooth);

    quantize_w4a4_act_fuse_lora(x, qact.act, qact.ascales, this->lora_down, qact.lora_act, this->smooth);

    debug("quantize.qact", qact.act);
    debug("quantize.ascales", qact.ascales);
    debug("quantize.lora_act", qact.lora_act);
#else 
    static const half one = 1.0;
    static const half zero = 0.0;

    nvtxRangePushA("LoraDown");

    // lora_down: [M, IC] * [IC, R] => [M, R]
    // cublas view: [R, IC] * [IC, M]
    checkCUBLAS(cublasHgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        this->lora_rank, M, this->in_features,
        &one,
        lora_down.data_ptr<half>(),
        this->lora_rank,
        x.data_ptr<half>(),
        this->in_features,
        &zero, 
        qact.lora_act.data_ptr<half>(),
        this->lora_rank));

    nvtxRangePop();

    quantize_w4a4_act(x, qact.act, qact.ascales);

#endif

    return qact;
}
