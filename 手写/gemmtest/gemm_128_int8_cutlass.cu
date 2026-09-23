#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/device_memory.h>

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// INT8专用分块 Block<64,64,16> MMA m16n8k16
using GemmShape = cutlass::gemm::GemmShape<64, 64, 16, 16, 16, 16, 8>;

using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    16, // 每线程输出16个int32
    ElementAcc,
    ElementAcc
>;

using GemmKernel = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm86,
    GemmShape,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    Epilogue
>;

int main() {
    const int M = 128;
    const int N = 128;
    const int K = 128;

    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> C_dev({M, N});
    cutlass::HostTensor<ElementC, LayoutC> C_ref({M, N});

    A.random_uniform(-127, 127);
    B.random_uniform(-127, 127);
    C_dev.fill(0);

    A.sync_device();
    B.sync_device();
    C_dev.sync_device();

    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        A.device_ref(),
        B.device_ref(),
        C_dev.device_ref(),
        C_dev.device_ref(),
        cutlass::epilogue::thread::LinearCombinationParams{1, 0}
    };

    GemmKernel gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("INT8 GEMM 配置失败\n");
        return -1;
    }

    size_t workspace = gemm_op.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> buf(workspace);
    status = gemm_op(args, buf.get());
    cudaDeviceSynchronize();

    C_dev.sync_host();

    cutlass::reference::host::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc
    >(
        cutlass::gemm::GemmCoord{M,N,K},
        A.host_ref(), B.host_ref(),
        C_ref.host_ref(), C_ref.host_ref(),
        {1,0}
    );

    bool ok = cutlass::util::compare_tensor(C_dev.host_view(), C_ref.host_view());
    printf("INT8测试: %s\n", ok ? "PASS" : "FAIL");
    return 0;
}