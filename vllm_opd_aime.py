import json
import multiprocessing as mp
import os
import tempfile

from transformers import AutoTokenizer
from tqdm import tqdm
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


MATH_QUERY_TEMPLATE = """{Question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.""".strip()


def _visible_devices():
    """Device ids this process is allowed to use, as seen by the parent."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def batch_generate_vllm(prompts, model_path, generation_config, system_prompt, batch_size,
                        tensor_parallel_size=1, seed=42, show_progress=True):
    # vLLM is imported lazily so that spawned data-parallel workers can pin
    # CUDA_VISIBLE_DEVICES *before* any CUDA context is created.
    from vllm import LLM, SamplingParams

    responses = []

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')

    # 初始化 Sampling 参数（与训练日志对齐：R1-Distill 官方推荐 T=0.6, top_p=0.95）
    max_model_len = generation_config.get("max_model_len", 16384)
    sampling_params = SamplingParams(
        temperature=generation_config.get("temperature", 0.6),
        top_k=generation_config.get("top_k", -1),
        top_p=generation_config.get("top_p", 0.95),
        max_tokens=generation_config.get("max_new_tokens", max_model_len),
    )

    # 加载模型（与训练侧一致：bfloat16、enable_prefix_caching）
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )

    for i in tqdm(range(0, len(prompts), batch_size), disable=not show_progress):
        batch_prompts = prompts[i:i + batch_size]

        # 构造 chat 格式输入 —— R1-Distill 不使用 system prompt，chat template 会自动补 <think>\n
        batch_messages = [tokenizer.apply_chat_template(
                                                      [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=prompt)}],
                                                      tokenize=False,
                                                      add_generation_prompt=True) for prompt in batch_prompts]

        # 执行批量推理
        outputs = llm.generate(batch_messages, sampling_params, use_tqdm=show_progress)

        # 打印结果
        for prompt, output in zip(batch_prompts, outputs):
            responses.append(output.outputs[0].text)
    return responses


def _dp_worker(rank, devices, shard_path, indices, prompts, model_path, generation_config,
               system_prompt, batch_size, tensor_parallel_size, seed):
    """One data-parallel shard: pin GPUs, generate, dump {index: response} to shard_path."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    # Keep each shard's own TP group from clashing on the same rendezvous port.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    responses = batch_generate_vllm(
        prompts,
        model_path,
        generation_config,
        system_prompt,
        batch_size=batch_size or len(prompts),
        tensor_parallel_size=tensor_parallel_size,
        # Distinct seeds so identical prompts in different shards don't produce
        # identical samples (avg@32 needs the samples to be independent).
        seed=seed + rank,
        show_progress=(rank == 0),
    )

    with open(shard_path, "w", encoding="utf-8") as f:
        for idx, response in zip(indices, responses):
            f.write(json.dumps({"index": idx, "response": response}, ensure_ascii=False) + "\n")


def data_parallel_generate(prompts, model_path, generation_config, system_prompt, batch_size,
                           data_parallel_size, tensor_parallel_size=1, seed=42):
    """Shard prompts across `data_parallel_size` independent vLLM instances."""
    if not prompts:
        return []

    devices = _visible_devices()
    needed = data_parallel_size * tensor_parallel_size
    if len(devices) < needed:
        raise RuntimeError(
            f"data_parallel_size={data_parallel_size} x tensor_parallel_size={tensor_parallel_size} "
            f"needs {needed} GPUs, but only {len(devices)} are visible: {devices}"
        )

    if data_parallel_size == 1:
        return batch_generate_vllm(
            prompts, model_path, generation_config, system_prompt,
            batch_size=batch_size or len(prompts),
            tensor_parallel_size=tensor_parallel_size, seed=seed,
        )

    data_parallel_size = min(data_parallel_size, len(prompts))
    # Strided (round-robin) sharding, not contiguous: prompts arrive as
    # num_generation copies of each problem back-to-back, so contiguous chunks
    # would hand one worker all the samples of a single (possibly very long)
    # problem and leave the rest idle.
    shards = [list(range(rank, len(prompts), data_parallel_size)) for rank in range(data_parallel_size)]

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="vllm_dp_") as tmpdir:
        procs = []
        shard_paths = []
        for rank, indices in enumerate(shards):
            if not indices:
                # Fewer prompts than shards; don't pay for a model load to do nothing.
                continue
            shard_path = os.path.join(tmpdir, f"shard_{rank}.jsonl")
            shard_paths.append(shard_path)
            proc = ctx.Process(
                target=_dp_worker,
                args=(
                    rank,
                    devices[rank * tensor_parallel_size:(rank + 1) * tensor_parallel_size],
                    shard_path,
                    indices,
                    [prompts[i] for i in indices],
                    model_path,
                    generation_config,
                    system_prompt,
                    batch_size,
                    tensor_parallel_size,
                    seed,
                ),
            )
            proc.start()
            procs.append((rank, proc))

        for rank, proc in procs:
            proc.join()
            if proc.exitcode != 0:
                for _, other in procs:
                    if other.is_alive():
                        other.terminate()
                raise RuntimeError(f"data-parallel worker rank={rank} exited with code {proc.exitcode}")

        responses = [None] * len(prompts)
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    responses[row["index"]] = row["response"]

    missing = [i for i, r in enumerate(responses) if r is None]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts came back without a response (e.g. index {missing[0]})")
    return responses


def predict(test_model_path, test_file, output_path, batch_size=None, tensor_parallel_size=1,
            data_parallel_size=1, seed=42, max_model_len=16384, num_generation=32):
    # 生成配置 —— R1-Distill 官方推荐 temperature=0.6 / top_p=0.95 / 不使用 top_k；
    # max_model_len 应与训练侧一致（16k 训练用 16384，DAPO-32k 训练用 32768）
    test_generation_config = {
        "max_model_len": max_model_len,
        "max_new_tokens": max_model_len,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "num_generation": num_generation,
    }

    # 系统提示词（R1-Distill 不使用 system prompt，此处保留占位，实际未使用）
    test_system_prompt = ""

    # 加载输入数据
    with open(test_file, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')

    test_prompts = [json.loads(row)['problem'] for row in rows]
    test_sol_ans = [(json.loads(row)['solution'], json.loads(row)['answer']) for row in rows]

    prompts = []
    for prompt in test_prompts:
        prompts += [prompt] * test_generation_config.get("num_generation", 1)
    sol_ans = []
    for data in test_sol_ans:
        sol_ans += [data] * test_generation_config.get("num_generation", 1)


    # 执行推理 —— batch_size 默认一次性全送，让 vLLM 自己做 continuous batching
    test_responses = data_parallel_generate(
        prompts,
        test_model_path,
        test_generation_config,
        test_system_prompt,
        batch_size=batch_size,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )
    save = []
    for problem, (sol, ans), response in zip(prompts, sol_ans, test_responses):
        save.append({'problem': problem, 'answer': ans, 'response': response})

    with open(output_path, 'w', encoding='utf-8') as f:
        for data in save:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def eval_metric(file_path):
    correct = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
    for row in rows:
        row = json.loads(row)
        answer = row['answer']
        response = row['response']

        response_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        reward = float(verify(response_parsed, answer))
        correct += reward
    print(f'acc: {round(correct / len(rows), 4)}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='HF checkpoint dir; omit to only score --output-path')
    parser.add_argument('--test-file', default='./aime_2024.jsonl')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--data-parallel-size', type=int, default=None,
                        help='Number of independent vLLM instances; defaults to '
                             'num_visible_gpus // tensor_parallel_size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-model-len', type=int, default=16384,
                        help='Context length; also caps generation. Match the '
                             'training setting (16384 for 16k runs, 32768 for DAPO-32k).')
    parser.add_argument('--num-generation', type=int, default=32,
                        help='Samples per problem (avg@N / pass@k base)')
    args = parser.parse_args()

    if args.model_path:
        data_parallel_size = args.data_parallel_size
        if data_parallel_size is None:
            data_parallel_size = max(1, len(_visible_devices()) // args.tensor_parallel_size)
        print(f'[eval] data_parallel_size={data_parallel_size} '
              f'tensor_parallel_size={args.tensor_parallel_size} '
              f'max_model_len={args.max_model_len} '
              f'num_generation={args.num_generation} '
              f'devices={_visible_devices()}')
        predict(args.model_path, args.test_file, args.output_path,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                data_parallel_size=data_parallel_size,
                seed=args.seed,
                max_model_len=args.max_model_len,
                num_generation=args.num_generation)
    eval_metric(args.output_path)
