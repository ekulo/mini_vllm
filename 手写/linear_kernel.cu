#include <torch/extension.h>

// GPU核函数声明
__global__ void gemm_kernel(float* out, const float* x, const float* w, int B, int H, int V);
__global__ void bias_add_kernel(float* out, const float* b, int B, int V);

// CPU宿主函数：Python调用入口，仅前向
torch::Tensor linear_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // 校验输入必须是cuda float32
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only support float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "only support float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "only support float32");

    int B = x.size(0);
    int H = x.size(1);
    int V = weight.size(0);

    // 分配输出显存
    auto out = torch::empty({B, V}, x.options());
    float* out_ptr = out.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.data_ptr<float>();

    // 线程配置
    const int block_size = 256;
    int total_elem = B * V;
    int grid_size = (total_elem + block_size - 1) / block_size;

    // 1. 矩阵乘 X @ W^T
    gemm_kernel<<<grid_size, block_size>>>(out_ptr, x_ptr, w_ptr, B, H, V);
    // 2. 加偏置
    bias_add_kernel<<<grid_size, block_size>>>(out_ptr, b_ptr, B, V);

    return out;
}

// 矩阵乘GPU核：每个线程计算1个输出元素 out[b,v] = sum_h x[b,h] * w[v,h]
__global__ void gemm_kernel(float* out, const float* x, const float* w, int B, int H, int V) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int total = B * V;
    if (idx >= total) return;

    int b = idx / V;
    int v = idx % V;

    float sum = 0.0f;
    int x_base = b * H;
    int w_base = v * H;
    for (int h = 0; h < H; h++) {
        sum += x[x_base + h] * w[w_base + h];
    }
    out[idx] = sum;
}

// 偏置相加GPU核：out[b,v] += b[v]
__global__ void bias_add_kernel(float* out, const float* b, int B, int V) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int total = B * V;
    if (idx >= total) return;

    int v = idx % V;
    out[idx] += b[v];
}

// Pybind 仅导出前向函数，删掉反向相关代码
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward_cuda, "Linear forward only CUDA kernel (inference only)");
}