
#include <torch/extension.h>



// 2. 先前置声明 GPU Kernel，Host forward 才能识别
__global__ void single_attn_fused_kernel(
    const float* q, const float* k, const float* v,
    const float* mask_f,
    float* out, float* attn_score_out, float* attn_weight_out,
    int B, int H, int N, int D, float scale
);


torch::Tensor single_attn_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    std::optional<torch::Tensor> mask_opt, float scale
){
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    auto B = q.size(0);
    auto H = q.size(1);
    auto N = q.size(2);
    auto D = q.size(3);

    auto attn_score = torch::empty({B,H,N,N}, q.options());
    auto attn_weight = torch::empty({B,H,N,N}, q.options());
    auto out = torch::empty_like(q);

    // ===================== Host 预处理 bool -> float mask =====================
    std::optional<torch::Tensor> mask_f_cache; // 持有tensor生命周期，防止data_ptr悬空
    const float* mask_f_ptr = nullptr;
    if(mask_opt.has_value()){
        // 转连续bool mask
        torch::Tensor mask_bool = mask_opt->contiguous().to(torch::kBool);
        // 预转换为float32，1.0f / 0.0f，GPU直接读取无需类型转换
        torch::Tensor mask_f = mask_bool.to(torch::kFloat32);
        mask_f_cache = mask_f; // 保存引用，函数内不会被销毁
        mask_f_ptr = mask_f.data_ptr<float>();
    }
    // ==========================================================================

    int block = 256;
    int grid = (B*H*N*N + block - 1) / block;
    single_attn_fused_kernel<<<grid, block>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        mask_f_ptr, // 传入预转好的float掩码指针
        out.data_ptr<float>(), attn_score.data_ptr<float>(), attn_weight.data_ptr<float>(),
        B, H, N, D, scale
    );
    return out;
}

__global__ void single_attn_fused_kernel(
    const float* q, const float* k, const float* v,
    const float* mask_f,  // 输入：预转换float掩码 1.0/0.0，nullptr代表无mask
    float* out, float* attn_score_out, float* attn_weight_out,
    int B, int H, int N, int D, float scale
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * N;
    if(idx >= total) return;

    // 解码索引 b, head, query token i, key token j
    int b = idx / (H * N * N);
    int rem1 = idx % (H * N * N);
    int h = rem1 / (N * N);
    int rem2 = rem1 % (N * N);
    int i = rem2 / N;
    int j = rem2 % N;

    // Step1 QK点积
    float dot = 0.0f;
    for(int d = 0; d < D; d++){
        long q_off = ((long)b * H * N + (long)h * N + i) * D + d;
        long k_off = ((long)b * H * N + (long)h * N + j) * D + d;
        dot += q[q_off] * k[k_off];
    }
    float score_raw = dot / scale;

    // Step2 无分支掩码计算，无if跳转，warp线程指令完全统一
    float m = 1.0f;
    // mask_f == nullptr 是全局统一判断：所有线程条件相同，无发散
    if(mask_f != nullptr){
        long mask_off = ((long)b * N + i) * N + j;
        m = mask_f[mask_off];
    }
    // 纯算术替换 if(!m) score = -1e9f
    float score = score_raw * m - (1.0f - m) * 1e9f;
    attn_score_out[idx] = score;

    // Step3 Softmax（原有逻辑不变，仅示例保留）
    float max_val = -1e10f;
    for(int jj = 0; jj < N; jj++){
        long jdx = ((long)b * H * N + (long)h * N + i) * N + jj;
        max_val = max(max_val, attn_score_out[jdx]);
    }
    float exp_sum = 0.0f;
    for(int jj = 0; jj < N; jj++){
        long jdx = ((long)b * H * N + (long)h * N + i) * N + jj;
        exp_sum += exp(attn_score_out[jdx] - max_val);
    }
    float weight = exp(score - max_val) / exp_sum;
    attn_weight_out[idx] = weight;

    // Step4 Attention @ V
    float y_val = 0.0f;
    for(int d = 0; d < D; d++){
        long v_off = ((long)b * H * N + (long)h * N + j) * D + d;
        y_val += weight * v[v_off];
    }
    long out_off = ((long)b * H * N + (long)h * N + i) * D + (threadIdx.x % D);
    out[out_off] = y_val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &single_attn_forward, "fused attn forward");
}