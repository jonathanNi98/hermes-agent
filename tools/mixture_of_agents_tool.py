#!/usr/bin/env python3
"""
# ============================================================
# Mixture-of-Agents Tool —— 多 LLM 协作 + 聚合(arXiv:2406.04692)
# ============================================================
# 1.1 本文件做什么
# ------------------------------------------------------------
#   把"一道难题"用多层 LLM 协作解决,论文方法叫 MoA:
#     - Layer 1:多个 reference model 并行独立作答(发散)
#     - Layer 2:aggregator model 读所有 reference 答案,综合出最终答案(收敛)
#
# 1.2 为什么这么设计(论文核心 insight)
# ------------------------------------------------------------
#   不同 LLM 在不同问题上有各自擅长 + 盲区;
#   把多个模型独立答案"喂"给一个强模型综合,通常比任何单模型都好。
#   (论文里的"叠 buff"效应 —— 类似 LLM ensemble)
#
# 1.3 文件组织
# ------------------------------------------------------------
#   1.x 模块头 + 配置常量
#   2.x prompt 构造工具
#   3.x reference model 调用(含 retry / 错误处理)
#   4.x aggregator model 调用
#   5.x 主入口 mixture_of_agents_tool
#   6.x 辅助函数(requirements check / config getter)
#   7.x __main__ 演示块
#   8.x Schema + registry 注册
# ============================================================
# 论文:"Mixture-of-Agents Enhances Large Language Model Capabilities"
#       Junlin Wang et al., arXiv:2406.04692v1
#
# 适用场景(只用在难题上,普通题不要用 —— 5 次 API call 太贵):
#   - 复杂数学证明 / 计算
#   - 高级算法设计
#   - 多步分析推理
#   - 需要多元领域知识的问题
#   - 单模型表现出局限的任务
#
# 模型(走 OpenRouter):
#   - Reference Models:claude-opus-4.6, gemini-2.5-pro, gpt-5.4-pro, deepseek-v3.2
#   - Aggregator Model: claude-opus-4.6(综合能力最强)
# ============================================================
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from tools.openrouter_client import get_async_client as _get_openrouter_client, check_api_key as check_openrouter_api_key
from agent.auxiliary_client import extract_content_or_reasoning
from tools.debug_helpers import DebugSession
import sys

logger = logging.getLogger(__name__)


# ===========================================================================
# 1.2 配置常量(改这里调整 MoA 行为,不用动函数体)
# ===========================================================================
# 1.2.1 REFERENCE_MODELS —— Layer 1 并行跑的模型列表
# ---------------------------------------------------------------------------
# 要求:尽量多元化(不同厂商、不同擅长),不要全是同质化模型
# 论文建议 3-6 个,多了边际收益递减
REFERENCE_MODELS = [
    "anthropic/claude-opus-4.6",
    "google/gemini-2.5-pro",
    "openai/gpt-5.4-pro",
    "deepseek/deepseek-v3.2",
]

# 1.2.2 AGGREGATOR_MODEL —— Layer 2 综合所有答案的模型
# ---------------------------------------------------------------------------
# 论文建议用最强的综合模型(Claude Opus 通常表现最好)
# 如果换便宜模型,综合质量会下降
AGGREGATOR_MODEL = "anthropic/claude-opus-4.6"

# 1.2.3 温度(reference 高一点保多样性,aggregator 低一点保稳定)
REFERENCE_TEMPERATURE = 0.6  # Balanced creativity for diverse perspectives
AGGREGATOR_TEMPERATURE = 0.4  # Focused synthesis for consistency

# 1.2.4 MIN_SUCCESSFUL_REFERENCES
# ---------------------------------------------------------------------------
# 至少要有几个 reference 成功,才能继续跑 aggregator
# 设 1 = 容忍所有 reference 都挂,只要 1 个成功就继续(降级方案)
# 设更高 = 更严格,但有 reference 全挂时整个 tool 直接 fail
MIN_SUCCESSFUL_REFERENCES = 1  # Minimum successful reference models needed to proceed

# 1.2.5 AGGREGATOR_SYSTEM_PROMPT —— 论文原版
# ---------------------------------------------------------------------------
# 强调:
#   - "批判性评估"(不要盲信参考)
#   - "不要直接复述"(要 refine)
#   - "结构化 / 高质量"
# 这是论文里反复验证有效的 prompt 模板,改坏容易掉质量
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

# 1.2.6 DebugSession —— 可选 debug 日志
# ---------------------------------------------------------------------------
# 设 MOA_TOOLS_DEBUG=true 启用,日志写到 ./logs/moa_tools_debug_<uuid>.json
# 包含每次调用的参数 / 成功失败的 reference 数 / 处理时间等
_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")


# ===========================================================================
# 2. _construct_aggregator_prompt —— 把 reference 答案拼到 aggregator prompt 里
# ===========================================================================
# 2.1 干什么
# ---------------------------------------------------------------------------
#   把基础 system_prompt + 编号列表形式的 reference 响应拼起来
#   编号 "1. xxx\n2. yyy" 让 aggregator 明确知道有几条参考
def _construct_aggregator_prompt(system_prompt: str, responses: List[str]) -> str:
    """
    Construct the final system prompt for the aggregator including all model responses.

    Args:
        system_prompt (str): Base system prompt for aggregation
        responses (List[str]): List of responses from reference models

    Returns:
        str: Complete system prompt with enumerated responses
    """
    response_text = "\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
    return f"{system_prompt}\n\n{response_text}"


# ===========================================================================
# 3. _run_reference_model_safe —— 跑一个 reference model + 重试 + 错误兜底
# ===========================================================================
# 3.1 设计要点
# ---------------------------------------------------------------------------
#   - 每个 reference 独立失败处理(不会因为某个模型挂了影响其他)
#   - 6 次重试 + 指数退避(2/4/8/16/32/60s,上限 60s)
#   - 空响应(reasoning-only)也当作失败 → 进重试
#   - 不同错误分类 log(invalid / rate limit / unknown)
#   - GPT 模型特殊处理:不支持自定义 temperature 参数(会被 400)
#
# 3.2 返值约定
# ---------------------------------------------------------------------------
#   (model_name, content_or_error_msg, success_bool)
#   - success=True → content 是真实响应
#   - success=False → content 是错误消息字符串
#
# 3.3 为什么需要指数退避
# ---------------------------------------------------------------------------
#   rate limit 通常是 429 + 重试窗口 5-60s,固定间隔会被持续 rate limit
#   指数退避让 server 有时间"消化"压力,也避免 thundering herd
async def _run_reference_model_safe(
    model: str,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 6
) -> tuple[str, str, bool]:
    """
    Run a single reference model with retry logic and graceful failure handling.

    Args:
        model (str): Model identifier to use
        user_prompt (str): The user's query
        temperature (float): Sampling temperature for response generation
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts

    Returns:
        tuple[str, str, bool]: (model_name, response_content_or_error, success_flag)
    """
    for attempt in range(max_retries):
        try:
            logger.info("Querying %s (attempt %s/%s)", model, attempt + 1, max_retries)

            # Build parameters for the API call
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
                "extra_body": {
                    "reasoning": {
                        "enabled": True,
                        "effort": "xhigh"
                    }
                }
            }

            # GPT models (especially gpt-4o-mini) don't support custom temperature values
            # Only include temperature for non-GPT models
            if not model.lower().startswith('gpt-'):
                api_params["temperature"] = temperature

            response = await _get_openrouter_client().chat.completions.create(**api_params)

            content = extract_content_or_reasoning(response)
            if not content:
                # Reasoning-only response — let the retry loop handle it
                logger.warning("%s returned empty content (attempt %s/%s), retrying", model, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
            logger.info("%s responded (%s characters)", model, len(content))
            return model, content, True

        except Exception as e:
            error_str = str(e)
            # Keep retry-path logging concise; full tracebacks are reserved for
            # terminal failure paths so long-running MoA retries don't flood logs.
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", model, attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", model, attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", model, attempt + 1, error_str)

            if attempt < max_retries - 1:
                # Exponential backoff for rate limiting: 2s, 4s, 8s, 16s, 32s, 60s
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{model} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return model, error_msg, False


# ===========================================================================
# 4. _run_aggregator_model —— 跑 aggregator 模型做最终综合
# ===========================================================================
# 4.1 与 _run_reference_model_safe 的区别
# ---------------------------------------------------------------------------
#   - 没有 reference 那么多重试(只重试 1 次,因为 empty 很少见)
#   - 用 system + user 双 message(喂入 reference 答案)
#   - 温度更低(0.4 vs 0.6) → 综合更稳定
#
# 4.2 为什么 system + user 而非只 user
# ---------------------------------------------------------------------------
#   - system:aggregator 的人格设定("你批判性综合"+ reference 列表)
#   - user:原始问题(让模型始终围绕原问题综合)
async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: int = None
) -> str:
    """
    Run the aggregator model to synthesize the final response.

    Args:
        system_prompt (str): System prompt with all reference responses
        user_prompt (str): Original user query
        temperature (float): Focused temperature for consistent aggregation
        max_tokens (int): Maximum tokens in final response

    Returns:
        str: Synthesized final response
    """
    logger.info("Running aggregator model: %s", AGGREGATOR_MODEL)

    # Build parameters for the API call
    api_params = {
        "model": AGGREGATOR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "extra_body": {
            "reasoning": {
                "enabled": True,
                "effort": "xhigh"
            }
        }
    }

    # GPT models (especially gpt-4o-mini) don't support custom temperature values
    # Only include temperature for non-GPT models
    if not AGGREGATOR_MODEL.lower().startswith('gpt-'):
        api_params["temperature"] = temperature

    response = await _get_openrouter_client().chat.completions.create(**api_params)

    content = extract_content_or_reasoning(response)

    # Retry once on empty content (reasoning-only response)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await _get_openrouter_client().chat.completions.create(**api_params)
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%s characters)", len(content))
    return content


# ===========================================================================
# 5. mixture_of_agents_tool —— 主入口(LLM 调的工具实际就是它)
# ===========================================================================
# 5.1 完整流程
# ---------------------------------------------------------------------------
#   1) 校验 API key 有没有
#   2) 解析 reference_models / aggregator_model 参数(可覆盖默认)
#   3) Layer 1:asyncio.gather 并行跑所有 reference
#   4) 分类成功/失败的 reference
#   5) 校验"至少 N 个成功" → 否则 raise
#   6) Layer 2:拼 aggregator prompt → 跑 aggregator
#   7) 算处理时间 + 写 debug 日志
#   8) 返 JSON 字符串(给 LLM 看)
#
# 5.2 为什么 asyncio.gather
# ---------------------------------------------------------------------------
#   4 个 reference model 是**独立**的,顺序无关
#   并发跑 → 总耗时 ≈ 最慢那个,而不是 4 个的累加
#   (实际 4 个 ~10s 总耗时,串行要 40s+)
#
# 5.3 返 JSON 格式
# ---------------------------------------------------------------------------
#   {"success": bool, "response": str, "models_used": {...}, "error"?: str}
#   - success: 是否真的产出聚合结果
#   - response: 聚合后的最终响应(给 LLM 看的内容)
#   - models_used: 用了哪些模型(可观测性)
#   - error: 失败时的错误消息
async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None
) -> str:
    """
    Process a complex query using the Mixture-of-Agents methodology.
    
    This tool leverages multiple frontier language models to collaboratively solve
    extremely difficult problems requiring intense reasoning. It's particularly
    effective for:
    - Complex mathematical proofs and calculations
    - Advanced coding problems and algorithm design
    - Multi-step analytical reasoning tasks
    - Problems requiring diverse domain expertise
    - Tasks where single models show limitations
    
    The MoA approach uses a fixed 2-layer architecture:
    1. Layer 1: Multiple reference models generate diverse responses in parallel (temp=0.6)
    2. Layer 2: Aggregator model synthesizes the best elements into final response (temp=0.4)
    
    Args:
        user_prompt (str): The complex query or problem to solve
        reference_models (Optional[List[str]]): Custom reference models to use
        aggregator_model (Optional[str]): Custom aggregator model to use
    
    Returns:
        str: JSON string containing the MoA results with the following structure:
             {
                 "success": bool,
                 "response": str,
                 "models_used": {
                     "reference_models": List[str],
                     "aggregator_model": str
                 },
                 "processing_time": float
             }
    
    Raises:
        Exception: If MoA processing fails or API key is not set
    """
    start_time = datetime.datetime.now()
    
    debug_call_data = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": reference_models or REFERENCE_MODELS,
            "aggregator_model": aggregator_model or AGGREGATOR_MODEL,
            "reference_temperature": REFERENCE_TEMPERATURE,
            "aggregator_temperature": AGGREGATOR_TEMPERATURE,
            "min_successful_references": MIN_SUCCESSFUL_REFERENCES
        },
        "error": None,
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "models_used": {}
    }
    
    try:
        logger.info("Starting Mixture-of-Agents processing...")
        logger.info("Query: %s", user_prompt[:100])
        
        # Validate API key availability
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Use provided models or defaults
        ref_models = reference_models or REFERENCE_MODELS
        agg_model = aggregator_model or AGGREGATOR_MODEL
        
        logger.info("Using %s reference models in 2-layer MoA architecture", len(ref_models))
        
        # Layer 1: Generate diverse responses from reference models (with failure handling)
        logger.info("Layer 1: Generating reference responses...")
        model_results = await asyncio.gather(*[
            _run_reference_model_safe(model, user_prompt, REFERENCE_TEMPERATURE)
            for model in ref_models
        ])
        
        # Separate successful and failed responses
        successful_responses = []
        failed_models = []
        
        for model_name, content, success in model_results:
            if success:
                successful_responses.append(content)
            else:
                failed_models.append(model_name)
        
        successful_count = len(successful_responses)
        failed_count = len(failed_models)
        
        logger.info("Reference model results: %s successful, %s failed", successful_count, failed_count)
        
        if failed_models:
            logger.warning("Failed models: %s", ', '.join(failed_models))
        
        # Check if we have enough successful responses to proceed
        if successful_count < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(f"Insufficient successful reference models ({successful_count}/{len(ref_models)}). Need at least {MIN_SUCCESSFUL_REFERENCES} successful responses.")
        
        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models
        
        # Layer 2: Aggregate responses using the aggregator model
        logger.info("Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, 
            successful_responses
        )
        
        final_response = await _run_aggregator_model(
            aggregator_system_prompt,
            user_prompt,
            AGGREGATOR_TEMPERATURE
        )
        
        # Calculate processing time
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info("MoA processing completed in %.2f seconds", processing_time)
        
        # Prepare successful response (only final aggregated result, minimal fields)
        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model
            }
        }
        
        debug_call_data["success"] = True
        debug_call_data["final_response_length"] = len(final_response)
        debug_call_data["processing_time_seconds"] = processing_time
        debug_call_data["models_used"] = result["models_used"]
        
        # Log debug information
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in MoA processing: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Calculate processing time even for errors
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare error response (minimal fields)
        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": reference_models or REFERENCE_MODELS,
                "aggregator_model": aggregator_model or AGGREGATOR_MODEL
            },
            "error": error_msg
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)


# ===========================================================================
# 6. 辅助函数:requirements check + config getter
# ===========================================================================
# 6.1 check_moa_requirements —— 注册时 check_fn 用
# ---------------------------------------------------------------------------
# 检查 OpenRouter API key 有没有(没 key 直接不能跑)
# 返 True/False → registry 用这个决定要不要把 tool 暴露给 LLM
def check_moa_requirements() -> bool:
    """
    Check if all requirements for MoA tools are met.

    Returns:
        bool: True if requirements are met, False otherwise
    """
    return check_openrouter_api_key()



# 6.2 get_moa_configuration —— 返当前配置(诊断 / debug 用)
# ---------------------------------------------------------------------------
# 包含:
#   - reference_models / aggregator_model:当前用哪些
#   - 各种温度 / 最小成功数
#   - failure_tolerance:"N/M models can fail"(字符串说明)
def get_moa_configuration() -> Dict[str, Any]:
    """
    Get the current MoA configuration settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration parameters
    """
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_model": AGGREGATOR_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": len(REFERENCE_MODELS),
        "failure_tolerance": f"{len(REFERENCE_MODELS) - MIN_SUCCESSFUL_REFERENCES}/{len(REFERENCE_MODELS)} models can fail"
    }


# ===========================================================================
# 7. __main__ —— 直接运行这个文件时的演示
# ===========================================================================
# 用法:`python mixture_of_agents_tool.py`
# 作用:
#   - 检查 API key
#   - 打当前配置(reference 模型数 / aggregator / 温度 / 失败容忍等)
#   - 打 debug mode 状态
#   - 打使用示例代码(给用户复制粘贴)
# 不是测试用例,只是给"我装好了,看看状态"的快速 sanity check
if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🤖 Mixture-of-Agents Tool Module")
    print("=" * 50)
    
    # Check if API key is available
    api_available = check_openrouter_api_key()
    
    if not api_available:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY='your-key-here'")
        print("Get API key at: https://openrouter.ai/")
        sys.exit(1)
    else:
        print("✅ OpenRouter API key found")
    
    print("🛠️  MoA tools ready for use!")
    
    # Show current configuration
    config = get_moa_configuration()
    print("\n⚙️  Current Configuration:")
    print(f"  🤖 Reference models ({len(config['reference_models'])}): {', '.join(config['reference_models'])}")
    print(f"  🧠 Aggregator model: {config['aggregator_model']}")
    print(f"  🌡️  Reference temperature: {config['reference_temperature']}")
    print(f"  🌡️  Aggregator temperature: {config['aggregator_temperature']}")
    print(f"  🛡️  Failure tolerance: {config['failure_tolerance']}")
    print(f"  📊 Minimum successful models: {config['min_successful_references']}")
    
    # Show debug mode status
    if _debug.active:
        print(f"\n🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/moa_tools_debug_{_debug.session_id}.json")
    else:
        print("\n🐛 Debug mode disabled (set MOA_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from mixture_of_agents_tool import mixture_of_agents_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      result = await mixture_of_agents_tool(")
    print("          user_prompt='Solve this complex mathematical proof...'")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nBest use cases:")
    print("  - Complex mathematical proofs and calculations")
    print("  - Advanced coding problems and algorithm design")
    print("  - Multi-step analytical reasoning tasks")
    print("  - Problems requiring diverse domain expertise")
    print("  - Tasks where single models show limitations")
    
    print("\nPerformance characteristics:")
    print("  - Higher latency due to multiple model calls")
    print("  - Significantly improved quality for complex tasks")
    print("  - Parallel processing for efficiency")
    print(f"  - Optimized temperatures: {REFERENCE_TEMPERATURE} for reference models, {AGGREGATOR_TEMPERATURE} for aggregation")
    print("  - Token-efficient: only returns final aggregated response")
    print("  - Resilient: continues with partial model failures")
    print("  - Configurable: easy to modify models and settings at top of file")
    print("  - State-of-the-art results on challenging benchmarks")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export MOA_TOOLS_DEBUG=true")
    print("  # Debug logs capture all MoA processing steps and metrics")
    print("  # Logs saved to: ./logs/moa_tools_debug_UUID.json")


# ===========================================================================
# 8. Schema + Registry —— 把 mixture_of_agents 装到 tool 注册表
# ===========================================================================
# 8.1 MOA_SCHEMA —— OpenAI function-calling 格式
# ---------------------------------------------------------------------------
# 只暴露一个参数 user_prompt(难题原文)
# reference_models / aggregator_model 用调用 mixture_of_agents_tool() 内部覆盖
# (避免 schema 里塞太多 enum 让 LLM 选错)
# description 强调"用 5 次 API call,只在难题上用,普通题不要用"
from tools.registry import registry

MOA_SCHEMA = {
    "name": "mixture_of_agents",
    "description": "Route a hard problem through multiple frontier LLMs collaboratively. Makes 5 API calls (4 reference models + 1 aggregator) with maximum reasoning effort — use sparingly for genuinely difficult problems. Best for: complex math, advanced algorithms, multi-step analytical reasoning, problems benefiting from diverse perspectives.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The complex query or problem to solve using multiple AI models. Should be a challenging problem that benefits from diverse perspectives and collaborative reasoning."
            }
        },
        "required": ["user_prompt"]
    }
}

# 8.2 registry.register —— 把 tool 真正装上
# ---------------------------------------------------------------------------
# 关键字段:
#   - name="mixture_of_agents":tool 标识
#   - toolset="moa":归 moa 工具集(用户可选择性启用)
#   - handler=lambda:把 args dict 翻译成本文件的 mixture_of_agents_tool(user_prompt=)
#   - check_fn=check_moa_requirements:没 API key 时不装
#   - requires_env=["OPENROUTER_API_KEY"]:硬依赖 env,启动时校验
#   - is_async=True:handler 是 async 的(registry 会 await 它)
#   - emoji="🧠":TUI 上显示
registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(user_prompt=args.get("user_prompt", "")),
    check_fn=check_moa_requirements,
    requires_env=["OPENROUTER_API_KEY"],
    is_async=True,
    emoji="🧠",
)
