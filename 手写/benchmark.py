from infra import Infra

import torch
import time

from torch.profiler import profile,record_function,ProfilerActivity
import cProfile,pstats


def speed_benchmark(model,prompt,max_token,repeat=10):
    with torch.no_grad():
        _=model.generate(prompt,max_token=max_token)
        if torch.cuda.is_available() :
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            _ = model.generate(prompt, max_token=max_token)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_cost = time.perf_counter() - start
        avg_latency = total_cost / repeat
        tps = max_token / avg_latency
        print("-" * 60)
        print(f"Prompt: {prompt}")
        print(f"单次平均耗时: {avg_latency:.4f} s")
        print(f"生成速度: {tps:.2f} token/s")
        print("-" * 60)


def run_profiler(model, prompt, max_token=25):
    with torch.no_grad():
        _ = model.generate(prompt, max_token=max_token)
        act = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            act.append(ProfilerActivity.CUDA)
        with profile(activities=act, record_shapes=True, with_stack=True) as prof:
            with record_function("llm_generate_full"):
                out = model.generate(prompt, max_token=max_token)
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
        prof.export_chrome_trace("generate_trace.json")
        print("火焰图导出完成 generate_trace.json")
        return out


def run_cprofile(model, prompt, max_token=25):
    prof_file = "python_profile.prof"
    cProfile.runctx(
        "model.generate(prompt, max_token=max_token)",
        globals=globals(),
        locals=locals(),
        filename=prof_file
    )
    stats = pstats.Stats(prof_file)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20)



if __name__ == "__main__":
    model = Infra()
    model = model.cuda()
    model.eval()
    test_prompt = "word_12 word_40 word_53 word_343 word_242"

    # 1. 测速
    speed_benchmark(model, test_prompt, max_token=25, repeat=5)
    # 2. 算子profile
    #run_profiler(model, test_prompt)
    # 3. python代码profile
    #run_cprofile(model, test_prompt)