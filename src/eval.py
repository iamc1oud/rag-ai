"""Retrieval + generation quality harness for RagChain.

Test set, the tuning iteration, and the ragas LLM-as-judge research/caveats:
README.md ("Evaluation" section).
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
# back-compat shim left in community. No version of langchain-community
# satisfies both packages at once. ChatVertexAI is never instantiated here
# (Ollama is the judge), so a placeholder module satisfies the import.
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


@dataclass
class TestCase:
    question: str
    expected_answer: str
    expected_pages: tuple[int, ...]  # any chunk from these pages counts as relevant


# 16 Q&A pairs against assets/Python Programming.pdf. expected_pages was
# determined by reading the source chapter each question targets, not by
# running retrieval first -- ground truth must be independent of the system
# being measured.
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


@dataclass
class RetrievalReport:
    k: int
    hit_rate: float
    mrr: float
    ranks: list[tuple[TestCase, int | None]]


@dataclass
class ComparisonReport:
    ks: tuple[int, ...]
    runs: list[RetrievalReport]
    rows: list[dict]  # {"question": str, "ranks": [rank_at_each_k]}


class RetrievalEvaluator:
    """Hit Rate@k and MRR -- a few lines of arithmetic, no eval library needed."""

    def __init__(self, store, test_set: list[TestCase] = TEST_SET):
        self.store = store
        self.test_set = test_set

    def _rank(self, case: TestCase, k: int) -> int | None:
        docs = self.store.similarity_search(case.question, k=k)
        pages = [d.metadata.get("page") for d in docs]
        return next((i for i, p in enumerate(pages, start=1) if p in case.expected_pages), None)

    def evaluate(self, k: int) -> RetrievalReport:
        ranks = [(case, self._rank(case, k)) for case in self.test_set]
        hits = [rank is not None for _, rank in ranks]
        return RetrievalReport(
            k=k,
            hit_rate=sum(hits) / len(hits),
            mrr=sum(1 / rank for _, rank in ranks if rank is not None) / len(ranks),
            ranks=ranks,
        )

    def compare(self, ks: tuple[int, ...] = (3, 5)) -> ComparisonReport:
        """Before/after table across k values -- the tuning iteration, in code."""
        runs = [self.evaluate(k) for k in ks]
        rows = [
            {"question": case.question, "ranks": [run.ranks[i][1] for run in runs]}
            for i, case in enumerate(self.test_set)
        ]
        return ComparisonReport(ks=ks, runs=runs, rows=rows)


class RagasJudge:
    """LLM-as-judge for faithfulness, answer relevancy, and context precision.

    Deliberately NOT settings.CHAT_MODEL (the generator) -- grading your own
    generations is ragas's documented self-preference bias. A different,
    smaller local model avoids that and avoids paying for judge calls on the
    cloud generator. What each metric measures and its limits: README.md.
    """

    MODEL = "ornith:9b"

    def __init__(self):
        client = AsyncOpenAI(base_url=f"{settings.OLLAMA_URL}/v1", api_key="ollama")
        llm = InstructorLLM(
            client=_get_instructor_client(client, "openai"),
            model=self.MODEL,
            provider="openai",
            # ragas's default max_tokens (1024) truncates ornith:9b mid-response
            # (it spends tokens on a <think> block before the structured JSON
            # verdict); llm_factory() can't override model_args without a kwarg
            # collision in ragas 0.4.3, so InstructorLLM is built directly here.
            model_args=InstructorModelArgs(max_tokens=4096),
        )
        embeddings = embedding_factory("openai", model=settings.EMBED_MODEL, client=client)
        self.faithfulness = Faithfulness(llm=llm)
        self.answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self.context_precision = ContextPrecisionWithoutReference(llm=llm)

    async def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        f, r, p = await asyncio.gather(
            self.faithfulness.ascore(user_input=question, response=answer,
                                     retrieved_contexts=contexts),
            self.answer_relevancy.ascore(user_input=question, response=answer),
            self.context_precision.ascore(user_input=question, response=answer,
                                          retrieved_contexts=contexts),
        )
        return {"faithfulness": f.value, "answer_relevancy": r.value, "context_precision": p.value}


@dataclass
class GenerationReport:
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    rows: list[dict]


class GenerationEvaluator:
    """Faithfulness/relevancy/precision for RagChain's generated answers."""

    def __init__(self, store, chain, judge: RagasJudge | None = None,
                 test_set: list[TestCase] = TEST_SET):
        self.store = store
        self.chain = chain
        self.judge = judge or RagasJudge()
        self.test_set = test_set

    async def _evaluate_async(self, k: int) -> GenerationReport:
        rows = []
        for case in self.test_set:
            docs = self.store.similarity_search(case.question, k=k)
            contexts = [d.page_content for d in docs]
            answer = self.chain.generate(case.question, docs)
            scores = await self.judge.score(case.question, answer, contexts)
            rows.append({"question": case.question, "answer": answer, **scores})

        n = len(rows)
        return GenerationReport(
            avg_faithfulness=sum(r["faithfulness"] for r in rows) / n,
            avg_answer_relevancy=sum(r["answer_relevancy"] for r in rows) / n,
            avg_context_precision=sum(r["context_precision"] for r in rows) / n,
            rows=rows,
        )

    def evaluate(self, k: int) -> GenerationReport:
        return asyncio.run(self._evaluate_async(k))


def print_report(retrieval: RetrievalReport, generation: GenerationReport | None = None) -> None:
    print(f"Hit Rate@{retrieval.k}: {retrieval.hit_rate:.3f}")
    print(f"MRR@{retrieval.k}:      {retrieval.mrr:.3f}")
    misses = [(case, rank) for case, rank in retrieval.ranks if rank is None or rank > 1]
    if misses:
        print(f"\n{len(misses)} question(s) not a rank-1 hit:")
        for case, rank in misses:
            print(f"  rank={rank!s:>4}  {case.question}")
    if generation:
        print(f"\navg faithfulness:       {generation.avg_faithfulness:.3f}")
        print(f"avg answer_relevancy:   {generation.avg_answer_relevancy:.3f}")
        print(f"avg context_precision:  {generation.avg_context_precision:.3f}")


def print_comparison(comparison: ComparisonReport) -> None:
    header = "".join(f"{'k=' + str(k):>12}" for k in comparison.ks)
    print(f"{'':30}{header}")
    print(f"{'Hit Rate':30}" + "".join(f"{r.hit_rate:>12.3f}" for r in comparison.runs))
    print(f"{'MRR':30}" + "".join(f"{r.mrr:>12.3f}" for r in comparison.runs))

    changed = [row for row in comparison.rows if len(set(row["ranks"])) > 1]
    if changed:
        print(f"\n{len(changed)} question(s) whose rank changed:")
        for row in changed:
            rank_str = " -> ".join(str(r) for r in row["ranks"])
            print(f"  {rank_str:20} {row['question']}")


def _self_check():
    class FakeStore:
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
    evaluator = RetrievalEvaluator(FakeStore(), cases)

    report = evaluator.evaluate(k=3)
    assert report.hit_rate == 2 / 3, report
    assert abs(report.mrr - (1 / 1 + 1 / 3 + 0) / 3) < 1e-9, report

    report_k1 = evaluator.evaluate(k=1)
    assert report_k1.hit_rate == 1 / 3, report_k1  # only the rank-1 hit survives

    comparison = evaluator.compare(ks=(1, 3))
    assert comparison.runs[0].hit_rate == 1 / 3
    assert comparison.runs[1].hit_rate == 2 / 3
    changed = [row for row in comparison.rows if len(set(row["ranks"])) > 1]
    assert [row["question"] for row in changed] == ["hit at rank 3"]
    assert changed[0]["ranks"] == [None, 3]

    print("ok (unit checks only; run eval.py for a live report)")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    if len(sys.argv) > 1 and sys.argv[1] == "live":
        from chain import RagChain
        from vectorstore import VectorStoreService

        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        store = VectorStoreService().load()
        retrieval = RetrievalEvaluator(store).evaluate(k=k)
        generation = None
        if "--no-ragas" not in sys.argv:
            generation = GenerationEvaluator(store, RagChain(store)).evaluate(k=k)
        print_report(retrieval, generation)
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        from vectorstore import VectorStoreService

        ks = tuple(int(x) for x in sys.argv[2:]) or (3, 5)
        store = VectorStoreService().load()
        print_comparison(RetrievalEvaluator(store).compare(ks=ks))
    else:
        _self_check()
