#include "common.h"
#include "Tensor.h"

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"

#include "cutlass/conv/device/direct_convolution.h"
#include "cutlass/conv/kernel/default_depthwise_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

// depthwise_Conv2d operation cutlass_sm80_tensorop_f16_s16x8x16fprop_analytic_f16_256x128_64x3_nhwc_align8

using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, 64>;

using FilterShape = cutlass::MatrixShape<3, 3>;

using ThreadblockShape =
    cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, 64, FilterShape::kCount>;

using WarpShape = cutlass::gemm::GemmShape<16, 64, FilterShape::kCount>;

using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80, // TODO
    ThreadblockShape,
    ThreadBlockOutputShape,
    FilterShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, float, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
    cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
        1,
        ThreadBlockOutputShape::kN,
        ThreadBlockOutputShape::kH,
        ThreadBlockOutputShape::kW>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedStrideDilation,
    cutlass::conv::StrideSupport::kFixed,
    cutlass::MatrixShape<1, 1>,
    cutlass::MatrixShape<1, 1>>::Kernel;

using DeviceKernel =
    typename cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

using UnderlyingKernel = typename DeviceKernel::UnderlyingKernel;
namespace {
    using TensorRefA = typename UnderlyingKernel::TensorRefA;
    using TensorRefB = typename UnderlyingKernel::TensorRefB;
    using TensorRefC = typename UnderlyingKernel::TensorRefC;
    using ElementCompute = typename UnderlyingKernel::EpilogueOutputOp::ElementCompute;
}

template <typename TensorRef, typename Element>
TensorRef get_tensor_ref(cutlass::Tensor4DCoord tensor_coord, Element *ptr) {
    cutlass::layout::TensorNHWC layout = cutlass::layout::TensorNHWC::packed(tensor_coord);
    TensorRef tensor_ref(ptr, layout);
    return tensor_ref;
}

static cutlass::Status depthwise_conv2d_kernel_run(cutlass::conv::Conv2dProblemSize *problem_size,
                                            UnderlyingKernel::ElementA *A, UnderlyingKernel::ElementB *B,
                                            UnderlyingKernel::ElementC *C, UnderlyingKernel::ElementC *D,
                                            ElementCompute alpha, ElementCompute beta, std::string split_k_mode,
                                            cudaStream_t stream, int device_id = 0)
{
    // create the tensor references
    cutlass::Tensor4DCoord tensor_coord_A = cutlass::conv::implicit_gemm_tensor_a_extent(
        cutlass::conv::Operator::kFprop, *problem_size);
    cutlass::Tensor4DCoord tensor_coord_B = cutlass::conv::implicit_gemm_tensor_b_extent(
        cutlass::conv::Operator::kFprop, *problem_size);
    cutlass::Tensor4DCoord tensor_coord_C = cutlass::conv::implicit_gemm_tensor_c_extent(
        cutlass::conv::Operator::kFprop, *problem_size);

    TensorRefA tensor_ref_A = get_tensor_ref<TensorRefA, UnderlyingKernel::ElementA>(tensor_coord_A, A);
    TensorRefB tensor_ref_B = get_tensor_ref<TensorRefB, UnderlyingKernel::ElementB>(tensor_coord_B, B);
    TensorRefC tensor_ref_C = get_tensor_ref<TensorRefC, UnderlyingKernel::ElementC>(tensor_coord_C, C);
    TensorRefC tensor_ref_D = get_tensor_ref<TensorRefC, UnderlyingKernel::ElementC>(tensor_coord_C, D);

    cutlass::conv::SplitKMode mode;
    if (split_k_mode == "serial") {
        mode = cutlass::conv::SplitKMode::kSerial;
    } else if (split_k_mode == "parallel") {
        mode = cutlass::conv::SplitKMode::kParallel;
    } else {
        throw std::runtime_error("Invalid split_k_mode: " + split_k_mode);
    }

    typename DeviceKernel::Arguments arguments{
        *problem_size,
        tensor_ref_A,
        tensor_ref_B,
        tensor_ref_C,
        tensor_ref_D,
        {alpha, beta},
        tensor_ref_B,
        // mode
    };

    DeviceKernel implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    BufferCUDA workspace(workspace_size);

    cutlass::Status status = implicit_gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = implicit_gemm_op.initialize(arguments, workspace.getPtr(), stream);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    //
    // Launch initialized CUTLASS kernel
    //
    status = implicit_gemm_op(stream);

    return status;
}

Tensor depthwise_conv2d_kernel(Tensor A, Tensor B) {
    int N, H, W, C_, K, C__, R, S, P, Q;
    N = A.size(0);
    H = A.size(1);
    W = A.size(2);
    C_ = A.size(3);

    // std::cout << A.dtype() << std::endl;
    // std::cout << "A: " << N << ", " << H << ", " << W << ", " << C_ << std::endl;
    // for (int h = 0; h < H; ++h)
    // {
    //     for (int w = 0; w < W; ++w)
    //         std::cout << A[0][h][w][0].item() << " ";
    //     std::cout << std::endl;
    // }

    K = B.size(0);
    R = B.size(1);
    S = B.size(2);
    C__ = B.size(3);

    // std::cout << B.dtype() << std::endl;
    // std::cout << "B: " << K << ", " << R << ", " << S << ", " << C__ << std::endl;
    // for (int h = 0; h < R; ++h)
    // {
    //     for (int w = 0; w < S; ++w)
    //         std::cout << B[0][h][w][0].item() << " ";
    //     std::cout << std::endl;
    // }

    cutlass::conv::Conv2dProblemSize problem_size(
        cutlass::Tensor4DCoord(N, H, W, C_),
        cutlass::Tensor4DCoord(K, R, S, C__),
        cutlass::Tensor4DCoord(1, 1, 1, 1),
        cutlass::MatrixCoord(1, 1),
        cutlass::MatrixCoord(1, 1),
        cutlass::conv::Mode::kCrossCorrelation,
        1,
        C_ // groups
    );

    P = problem_size.P;
    Q = problem_size.Q;

    // printf("P=%d Q=%d\n", P, Q);

    typename UnderlyingKernel::ElementC *ptrC = nullptr;

    Tensor D = Tensor::allocate({N, P, Q, K}, A.dtype(), A.device());

    auto stream = getCurrentCUDAStream();

    cutlass::Status status = depthwise_conv2d_kernel_run(
        &problem_size,
        reinterpret_cast<typename UnderlyingKernel::ElementA *>(A.data_ptr()),
        reinterpret_cast<typename UnderlyingKernel::ElementB *>(B.data_ptr()),
        ptrC,
        reinterpret_cast<typename UnderlyingKernel::ElementC *>(D.data_ptr()),
        1, 0,
        "serial", stream, B.device().idx);
    assert(status == cutlass::Status::kSuccess);

    return D;
}
