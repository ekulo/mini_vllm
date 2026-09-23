#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/util/device_memory.h>

// 数据类型定义 FP16 GEMM
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAcc = float;

// 行优先布局
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// 核心分块配置：Block<32,32,32> Warp<8,8> MMA m16n8k8
using GemmShape = cutlass::gemm::GemmShape<32, 32, 32, 8, 8, 16, 8>;

// 后处理：每线程输出4个元素，alpha*AB + beta*C
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    4,
    ElementAcc,
    ElementAcc
>;

// 完整GEMM内核定义，sm86适配RTX30/40/50系
using GemmKernel = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm86,
    GemmShape,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, // 双缓冲流水线 stage=2
    Epilogue
>;

int main() {
    // 固定128×128矩阵
    const int M = 128;
    const int N = 128;
    const int K = 128;

    // 主机张量分配
    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> C_dev({M, N});
    cutlass::HostTensor<ElementC, LayoutC> C_ref({M, N});

    // 填充随机值
    A.random_uniform(-1.0f, 1.0f);
    B.random_uniform(-1.0f, 1.0f);
    C_dev.fill(0);

    // 拷贝数据到GPU
    A.sync_device();
    B.sync_device();
    C_dev.sync_device();

    // GEMM参数 alpha=1, beta=0
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        A.device_ref(),
        B.device_ref(),
        C_dev.device_ref(),
        C_dev.device_ref(),
        cutlass::epilogue::thread::LinearCombinationParams{1.0f, 0.0f}
    };

    GemmKernel gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("GEMM配置校验失败\n");
        return -1;
    }

    // 分配工作空间
    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    // 执行GPU GEMM
    status = gemm_op(args, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        printf("GEMM运行失败\n");
        return -1;
    }
    cudaDeviceSynchronize();

    // 结果回拷CPU
    C_dev.sync_host();

    // CPU参考基准计算
    cutlass::reference::host::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAcc
    >(
        cutlass::gemm::GemmCoord{M, N, K},
        A.host_ref(),
        B.host_ref(),
        C_ref.host_ref(),
        C_ref.host_ref(),
        {1.0f, 0.0f}
    );

    // 精度校验
    bool pass = cutlass::util::compare_tensor(C_dev.host_view(), C_ref.host_view());
    printf("测试结果: %s\n", pass ? "PASS" : "FAIL");

    return 0;
}