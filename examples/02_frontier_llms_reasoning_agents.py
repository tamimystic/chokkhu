"""Example 2: Frontier LLMs, Linear Attention, MCTS Reasoning & Autonomous AI Agents."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.core.tensor import Tensor
from chokkhu.models.nlp import BPETokenizer, DeepSeekMLA, RoPE
from chokkhu.models.nlp.agents import ReActAgent, ToolRegistry
from chokkhu.models.nlp.reasoning import MonteCarloTreeSearchReasoning


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 2: FRONTIER LLMS, REASONING & AUTONOMOUS AGENTS")
    print("=" * 70)

    # 1. BPE Tokenization
    corpus = [
        "Chokkhu is a sovereign zero-heavy-dependency machine learning framework.",
        "Deep learning and transformer architectures built from pure NumPy mathematical formulations.",
        "Autonomous agents solve complex tasks using tool calling and reasoning search.",
    ]
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)
    sample_text = "Chokkhu autonomous agents reasoning."
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)
    print(
        f"[1] Tokenizer:\n  Original : {sample_text}\n  Token IDs: {token_ids}\n  Decoded  : {decoded_text}"
    )

    # 2. Rotary Position Embedding (RoPE) & DeepSeek MLA
    print(
        "\n[2] Transformer Building Blocks (RoPE & DeepSeek Multi-Head Latent Attention):"
    )
    head_dim = 16
    seq_len = 8
    rope = RoPE(dim=head_dim)
    q = np.random.randn(2, 4, seq_len, head_dim)
    q_rot = rope.apply_rope(q, seq_len=seq_len)
    print(f"  RoPE output tensor shape: {q_rot.shape}")

    mla = DeepSeekMLA(embed_dim=64, num_heads=4, head_dim=16, kv_latent_dim=16)
    x = Tensor(np.random.randn(2, seq_len, 64))
    mla_out = mla(x)
    print(f"  DeepSeek MLA output tensor shape: {mla_out.shape}")

    # 3. Autonomous AI Agent with Tool Registry
    print("\n[3] Autonomous ReAct Agent Execution:")
    tools = ToolRegistry()

    def calculator(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error: {e}"

    def stats_summary(numbers: str) -> str:
        arr = [float(x.strip()) for x in numbers.split(",") if x.strip()]
        return f"Mean={np.mean(arr):.2f}, Std={np.std(arr):.2f}, Count={len(arr)}"

    tools.register_func(
        name="calculator",
        description="Evaluates basic arithmetic expressions",
        func=calculator,
    )
    tools.register_func(
        name="stats_summary",
        description="Calculates mean and standard deviation of numbers",
        func=stats_summary,
    )

    agent = ReActAgent(tools=tools, max_iterations=4)
    prompt = "Calculate the average of 10, 20, 30, 40, 50 and then multiply by 2."
    response = agent.run(prompt)
    print(f"  Prompt : {prompt}")
    print(f"  Result : {response}")

    # 4. Test-Time Reasoning Search (MCTS)
    print("\n[4] Monte Carlo Tree Search (MCTS) Reasoning:")
    mcts = MonteCarloTreeSearchReasoning(c_puct=1.414)

    def step_generator(state: str):
        return [
            ("Formulate mathematical hypothesis", 0.6),
            ("Analyze statistical invariants", 0.4),
        ]

    def value_evaluator(state: str) -> float:
        return 0.85

    root_node = mcts.search(
        initial_prompt="Explore optimal learning trajectory",
        step_generator=step_generator,
        value_evaluator=value_evaluator,
        num_simulations=10,
    )
    print(
        f"  MCTS Search Root Visits: {root_node.visits}, Children: {list(root_node.children.keys())}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
