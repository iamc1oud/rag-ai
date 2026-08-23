"""Retrieval + generation quality harness for the RAG chain (chain.py).

Test set, the tuning iteration (failure -> root cause -> fix -> re-run), and
the ragas LLM-as-judge research/caveats: README.md ("Evaluation" section).
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass

from config import settings

# ---------------------------------------------------------------------------
# ragas 0.4.3 unconditionally imports langchain_community.chat_models.vertexai
# at import time. Current langchain-community (required by langchain-chroma
# 1.x, which requires langchain-core>=1.1) no longer ships that submodule --
# it moved to the standalone langchain-google-vertexai package with no
# back-compat shim left in community. Pinning community old enough to have it
# breaks chroma (needs langchain-core>=1.1; that old community needs <1.0),
# so there is no version combination that satisfies both packages' real
# requirements. We never instantiate ChatVertexAI here (Ollama is the judge),
# so a placeholder registered before the import satisfies ragas's import
# without needing real Vertex AI support. Real upstream incompatibility,
# verified by tracing both packages' actual code, not a guess.
_vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAIPlaceholder:
    pass


_vertexai_shim.ChatVertexAI = _ChatVertexAIPlaceholder
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_shim)

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings import embedding_factory  # noqa: E402
from ragas.llms.base import (  # noqa: E402
    InstructorLLM,
    InstructorModelArgs,
    _get_instructor_client,
)
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

# ragas's default max_tokens (1024, via InstructorModelArgs) truncates ornith:9b
# mid-response: it's a reasoning model that spends tokens on a <think> block
# before the structured JSON verdict, so the low default cuts it off before
# the answer. llm_factory() has no way to override model_args without a
# kwarg collision in ragas 0.4.3's InstructorAdapter (it always constructs
# its own InstructorModelArgs() and then re-passes **kwargs, so a caller's
# model_args collides). Building InstructorLLM directly sidesteps that.

# The ragas judge is deliberately NOT settings.CHAT_MODEL (the generator).
# Grading your own generations with the same model is a known ragas failure
# mode (self-preference bias -- a model rates its own phrasing/style higher
# than an equally correct answer worded differently). Using a different,
# smaller local model as judge avoids that and avoids paying for judge calls
# on the cloud model. See README for the fuller bias discussion.
JUDGE_MODEL = "ornith:9b"


@dataclass
class TestCase:
    question: str
    expected_answer: str  # short reference answer, for a human reading the report
    expected_pages: tuple[int, ...]  # any chunk from these pages counts as relevant


# 16 Q&A pairs against assets/Python Programming.pdf. expected_pages was
# determined by reading the source chapter each question targets (ingest.py
# output), not by running retrieval first -- ground truth has to be
# independent of the system being measured.
TEST_SET = [
    TestCase("What year was Python first released?", "1991", (21,)),
    TestCase("What is CPython?",
              "The reference implementation of Python, written in C.", (21,)),
    TestCase("What is the recommended basic editor for writing your first Python programs?",
              "IDLE", (30,)),
    TestCase("How do I open the Python console on Windows?",
              "Open the Command Prompt (cmd), e.g. via Windows+R, then type python.", (33,)),
    TestCase("What is the Python Standard Library?",
              "A large collection of built-in modules (file I/O, math, etc.) "
              "that ship with Python and don't need separate installation.", (45,)),
    TestCase("How do you print 'Hello World' in Python?",
              'print("Hello World!")', (40,)),
    TestCase("How do you write a basic For loop in Python?",
              "for i in range(1, 10): print(i)", (58,)),
    TestCase("What is a function in Python?",
              "A block of code that only runs when called; it can take "
              "parameters and return data.", (64,)),
    TestCase("How do you create a class in Python?",
              "With the `class` keyword, e.g. `class ClassName:`.", (70,)),
    TestCase("What function is used to open files in Python?",
              "open(), which takes a filename and a mode.", (78,)),
    TestCase("What are the two kinds of errors in Python?",
              "Syntax errors and exceptions.", (83,)),
    TestCase("What is PIP used for in Python?",
              "Installing and managing Python packages.", (87,)),
    TestCase("What is Anaconda?",
              "A Python Distribution package (not an editor); it includes Spyder.", (92,)),
    TestCase("What is Spyder?",
              "A Scientific PYthon Development EnviRonment -- a Python editor/IDE.", (99,)),
    TestCase("What is the Jupyter Notebook?",
              "An open-source web app for documents with live code, "
              "supporting 40+ languages including Python.", (116,)),
    TestCase("How do I plot a sine function in Python using matplotlib?",
              "Use numpy to generate x values, y = np.sin(x), then "
              "plt.plot(x, y) and plt.show().", (50, 51)),
]


def retrieve_pages(store, question: str, k: int) -> list[int]:
    docs = store.similarity_search(question, k=k)
    return [doc.metadata.get("page") for doc in docs]


def retrieval_metrics(store, test_set: list[TestCase], k: int) -> dict:
    """Hit Rate@k and MRR@k, plus the per-question ranks (for failure analysis).

    Hit Rate@k: fraction of questions where at least one expected page appears
    in the top k retrieved chunks.
    MRR: mean of 1/rank of the first expected-page hit (0 if none in top k).
    Both are a few lines of arithmetic -- no eval library needed for these two.
    """
    ranks = []
    for case in test_set:
        pages = retrieve_pages(store, case.question, k)
        rank = next(
            (i for i, p in enumerate(pages, start=1) if p in case.expected_pages), None
        )
        ranks.append(rank)

    hits = [r is not None for r in ranks]
    return {
        "k": k,
        "hit_rate": sum(hits) / len(hits),
        "mrr": sum(1 / r for r in ranks if r is not None) / len(ranks),
        "ranks": list(zip(test_set, ranks)),
    }


def _judge_clients():
    client = AsyncOpenAI(base_url=f"{settings.OLLAMA_URL}/v1", api_key="ollama")
    patched_client = _get_instructor_client(client, "openai")
    llm = InstructorLLM(
        client=patched_client,
        model=JUDGE_MODEL,
        provider="openai",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    embeddings = embedding_factory("openai", model=settings.EMBED_MODEL, client=client)
    return llm, embeddings


async def _generation_metrics_async(store, test_set: list[TestCase], k: int) -> dict:
    """Faithfulness, answer relevancy, and context precision, via ragas + Ollama.

    Faithfulness: does the answer only claim things the retrieved context
    supports? Answer relevancy: does the answer actually address the
    question? Context precision: are the top-ranked retrieved chunks the
    ones actually useful for answering? All three are LLM-as-judge metrics
    (see README for how ragas computes each and their known limitations).
    """
    import warnings

    from chain import build_chain, format_docs

    warnings.filterwarnings("ignore")
    llm, embeddings = _judge_clients()
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecisionWithoutReference(llm=llm)
    chain = build_chain()

    rows = []
    for case in test_set:
        docs = store.similarity_search(case.question, k=k)
        contexts = [d.page_content for d in docs]
        answer = chain.invoke({"context": format_docs(docs), "question": case.question})

        f, r, p = await asyncio.gather(
            faithfulness.ascore(
                user_input=case.question, response=answer, retrieved_contexts=contexts
            ),
            answer_relevancy.ascore(user_input=case.question, response=answer),
            context_precision.ascore(
                user_input=case.question, response=answer, retrieved_contexts=contexts
            ),
        )
        rows.append(
            {
                "question": case.question,
                "answer": answer,
                "faithfulness": f.value,
                "answer_relevancy": r.value,
                "context_precision": p.value,
            }
        )

        print(f"question: {case.question}")
        print(f"answer: {answer}")
        print(f"faithfulness: {f.value}")
        print(f"answer_relevancy: {r.value}")
        print(f"context_precision: {p.value}")
        print('--------')

    n = len(rows)
    return {
        "avg_faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "avg_answer_relevancy": sum(r["answer_relevancy"] for r in rows) / n,
        "avg_context_precision": sum(r["context_precision"] for r in rows) / n,
        "rows": rows,
    }


def generation_metrics(store, test_set: list[TestCase] = TEST_SET, k: int = 5) -> dict:
    return asyncio.run(_generation_metrics_async(store, test_set, k))


def compare_k(store, test_set: list[TestCase], ks: tuple[int, ...] = (3, 5)) -> dict:
    """Run retrieval_metrics at each k in `ks` and return a before/after table.

    This is the tuning iteration itself, in code: two eval runs plus a diff,
    not just numbers copied into README by hand. `python eval.py compare 3 5`
    reruns this against the live store and prints the same table.
    """
    runs = [retrieval_metrics(store, test_set, k=k) for k in ks]

    # per-question rank at each k, so a "before" miss that "after" fixes is
    # visible, not just the aggregate hit_rate/mrr moving
    rows = [
        {"question": case.question, "ranks": [run["ranks"][i][1] for run in runs]}
        for i, case in enumerate(test_set)
    ]

    return {"ks": ks, "runs": runs, "rows": rows}


def print_comparison(comparison: dict) -> None:
    ks = comparison["ks"]
    runs = comparison["runs"]
    header = "".join(f"{'k=' + str(k):>12}" for k in ks)
    print(f"{'':30}{header}")
    print(f"{'Hit Rate':30}" + "".join(f"{r['hit_rate']:>12.3f}" for r in runs))
    print(f"{'MRR':30}" + "".join(f"{r['mrr']:>12.3f}" for r in runs))

    changed = [row for row in comparison["rows"] if len(set(row["ranks"])) > 1]
    if changed:
        print(f"\n{len(changed)} question(s) whose rank changed:")
        for row in changed:
            rank_str = " -> ".join(str(r) for r in row["ranks"])
            print(f"  {rank_str:20} {row['question']}")


def print_report(retrieval: dict, generation: dict | None = None) -> None:
    print(f"Hit Rate@{retrieval['k']}: {retrieval['hit_rate']:.3f}")
    print(f"MRR@{retrieval['k']}:      {retrieval['mrr']:.3f}")
    misses = [(case, rank) for case, rank in retrieval["ranks"] if rank is None or rank > 1]
    if misses:
        print(f"\n{len(misses)} question(s) not a rank-1 hit:")
        for case, rank in misses:
            print(f"  rank={rank!s:>4}  {case.question}")
    if generation:
        print(f"\navg faithfulness:       {generation['avg_faithfulness']:.3f}")
        print(f"avg answer_relevancy:   {generation['avg_answer_relevancy']:.3f}")
        print(f"avg context_precision:  {generation['avg_context_precision']:.3f}")


def _self_check():
    class FakeStore:
        # question -> pages, in retrieved order, deterministic for the test
        _pages = {
            "hit at rank 1": [1, 2, 3],
            "hit at rank 3": [9, 9, 1],
            "miss": [9, 9, 9],
        }

        def similarity_search(self, question, k):
            from langchain_core.documents import Document

            return [Document(page_content="x", metadata={"page": p})
                    for p in self._pages[question][:k]]

    cases = [
        TestCase("hit at rank 1", "", (1,)),
        TestCase("hit at rank 3", "", (1,)),
        TestCase("miss", "", (1,)),
    ]
    m = retrieval_metrics(FakeStore(), cases, k=3)
    assert m["hit_rate"] == 2 / 3, m
    assert abs(m["mrr"] - (1 / 1 + 1 / 3 + 0) / 3) < 1e-9, m

    m_k1 = retrieval_metrics(FakeStore(), cases, k=1)
    assert m_k1["hit_rate"] == 1 / 3, m_k1  # only the rank-1 hit survives a smaller k

    comparison = compare_k(FakeStore(), cases, ks=(1, 3))
    assert comparison["runs"][0]["hit_rate"] == 1 / 3  # k=1
    assert comparison["runs"][1]["hit_rate"] == 2 / 3  # k=3
    changed = [row for row in comparison["rows"] if len(set(row["ranks"])) > 1]
    assert [row["question"] for row in changed] == ["hit at rank 3"]
    assert changed[0]["ranks"] == [None, 3]  # miss at k=1, rank 3 at k=3

    print("ok (unit checks only; run eval.py for a live report)")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    if len(sys.argv) > 1 and sys.argv[1] == "live":
        from vectorstore import load_vectorstore

        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        store = load_vectorstore()
        retrieval = retrieval_metrics(store, TEST_SET, k=k)
        run_generation = "--no-ragas" not in sys.argv
        generation = generation_metrics(store, TEST_SET, k=k) if run_generation else None
        print_report(retrieval, generation)
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        from vectorstore import load_vectorstore

        ks = tuple(int(x) for x in sys.argv[2:]) or (3, 5)
        store = load_vectorstore()
        print_comparison(compare_k(store, TEST_SET, ks=ks))
    else:
        _self_check()
