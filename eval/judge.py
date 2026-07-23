"""Shared accuracy evaluation — LLM-as-a-Judge + BERTScore (evaluate.load,
baseline-rescaled). Single implementation used by eval/evaluate.py (offline benchmark)
and api/app.py (live dashboard) so the two never drift apart.

Judge mode is selected by JUDGE_MODE (default 'gemini'):
  gemini — one Gemini model grades every row with a single STRICT prompt. Chosen so
           the whole run is judged consistently (no mid-run judge swap) and by the
           strongest available model, while the strict rubric keeps it from inflating
           pass rates. The judge is disclosed per row as 'gemini_strict'.
  hf     — HF InferenceClient (the hackathon spec's designated judge) grades every row
           with the strict prompt; kept for spec-compliance / cross-checking. No silent
           mid-run fallback in either mode: a genuine judge error is raised, not turned
           into a lenient second opinion.
"""
import os
import warnings

from dotenv import load_dotenv
load_dotenv()

JUDGE_MODE     = os.getenv('JUDGE_MODE', 'gemini').lower()
HF_JUDGE_MODEL = os.getenv('HF_JUDGE_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')

_judge_client = None
_bertscore_metric = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        from huggingface_hub import InferenceClient
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise RuntimeError("HF_TOKEN environment variable is not set")
        _judge_client = InferenceClient(model=HF_JUDGE_MODEL, token=token)
    return _judge_client


# One strict rubric, used verbatim by whichever model is judging, so the model can change
# without the grading standard changing. Strict = the answer must actually answer THIS
# question and be consistent with the ground truth; partial, hedged, or off-question
# answers fail. Deliberately not "generous": lenient wording is what inflated pass rates.
STRICT_JUDGE_PROMPT = """You are a strict grader. Reply with exactly one word: PASS or FAIL.

PASS only if the system answer correctly answers the question AND its key facts agree
with the correct answer. Different wording or extra correct detail is fine.
FAIL if the answer is wrong, missing, incomplete on the main point, self-contradictory,
hedged into a non-answer, or answers a different question than the one asked.

Question: {q}
Correct answer: {correct}
System answer: {answer}

Verdict (PASS or FAIL):"""


def llm_judge(question: str, ground_truth: str, prediction: str) -> str:
    verdict, _ = llm_judge_with_source(question, ground_truth, prediction)
    return verdict


def _judge_hf(prompt: str) -> str:
    client = _get_judge_client()
    response = client.chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return response.choices[0].message.content.upper()


def _judge_gemini(prompt: str) -> str:
    from pipelines.utils import setup_gemini, gemini_generate
    client = setup_gemini()
    return gemini_generate(client, prompt, max_tokens=5).upper()


def llm_judge_with_source(question: str, ground_truth: str, prediction: str) -> tuple[str, str]:
    """Grade one answer and report which judge produced the verdict, for transparent
    reporting. A single model (JUDGE_MODE) grades the entire run with the one STRICT
    prompt -- no mid-run judge swap, no lenient fallback. Judge/model errors propagate
    (they are not converted into a PASS or into a softer second opinion)."""
    prompt = STRICT_JUDGE_PROMPT.format(q=question, correct=ground_truth, answer=prediction)
    if JUDGE_MODE == 'hf':
        return ('PASS' if 'PASS' in _judge_hf(prompt) else 'FAIL'), 'huggingface_strict'
    return ('PASS' if 'PASS' in _judge_gemini(prompt) else 'FAIL'), 'gemini_strict'


# Brief-mandated graded scale, judged blind to which pipeline produced the answer.
GRADED_JUDGE_PROMPT = """You are a careful grader. Reply with exactly one number: 3, 2, 1, or 0.

3 = fully correct: answers the question completely; all key facts agree with the correct answer.
2 = mostly correct: main point right, but a minor fact/member is missing or slightly off.
1 = partially correct: some relevant correct content, but the main point is incomplete or wrong.
0 = incorrect: wrong, empty, off-question, or contradicts the correct answer.

Question: {q}
Correct answer: {correct}
System answer: {answer}

Grade (3, 2, 1, or 0):"""


def graded_judge(question: str, ground_truth: str, prediction: str) -> int:
    """Brief-required graded answer-quality score (0-3), same judge model as strict."""
    prompt = GRADED_JUDGE_PROMPT.format(q=question, correct=ground_truth, answer=prediction)
    out = _judge_hf(prompt) if JUDGE_MODE == 'hf' else _judge_gemini(prompt)
    for ch in out:
        if ch in '0123':
            return int(ch)
    return 0


# Citation/evidence quality: does the evidence actually supplied to the generator
# support the answer's key claims? Blind to pipeline; evidence text is truncated
# to keep the judging call cheap.
CITATION_JUDGE_PROMPT = """You are auditing whether an answer is supported by the evidence that was supplied to the system that wrote it. Reply with exactly one number: 2, 1, or 0.

2 = supported: the key facts in the answer appear in (or follow directly from) the evidence.
1 = partially supported: some key facts are in the evidence, others are not.
0 = unsupported: the evidence does not contain the answer's key facts (or no evidence given yet the answer asserts facts).

Evidence supplied:
{evidence}

Question: {q}
Answer: {answer}

Score (2, 1, or 0):"""


def citation_judge(question: str, prediction: str, evidence: str) -> int:
    """Brief-required citation/evidence-quality score (0-2)."""
    if not (prediction or "").strip():
        return 0
    ev = (evidence or "(no evidence was supplied)")[:8000]
    prompt = CITATION_JUDGE_PROMPT.format(evidence=ev, q=question, answer=prediction)
    out = _judge_hf(prompt) if JUDGE_MODE == 'hf' else _judge_gemini(prompt)
    for ch in out:
        if ch in '012':
            return int(ch)
    return 0


def compute_bertscore(predictions: list, references: list) -> dict:
    """Real, per-hackathon-spec BERTScore: evaluate.load('bertscore') with
    rescale_with_baseline=True. Can be disabled on memory-constrained hosts via
    DISABLE_BERTSCORE=1 — the honest fallback is an explicit zero, never a fake score.
    """
    if os.getenv('DISABLE_BERTSCORE', '').lower() in ('1', 'true', 'yes'):
        return {'raw_f1': 0.0, 'rescaled_f1': 0.0, 'bonus_hit': False, 'disabled': True}

    global _bertscore_metric
    try:
        # transformers v5 removed PreTrainedTokenizerBase.build_inputs_with_special_tokens,
        # which bert-score (<=0.3.13, unmaintained) still calls -- without this shim the
        # metric crashes and the summary reports 0.0. The shim reproduces the standard
        # single/pair special-token layout ([CLS] a [SEP] / [CLS] a [SEP] b [SEP]).
        from transformers import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, 'build_inputs_with_special_tokens'):
            def _build_inputs_shim(self, token_ids_0, token_ids_1=None):
                cls = [self.cls_token_id] if self.cls_token_id is not None else []
                sep = [self.sep_token_id] if self.sep_token_id is not None else []
                out = cls + list(token_ids_0) + sep
                if token_ids_1:
                    out += list(token_ids_1) + sep
                return out
            PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_shim

        # Use the spec-mandated bert-score package DIRECTLY (bert-score==0.3.13).
        # NOTE: we deliberately avoid HF `import evaluate` here — running
        # `python eval/evaluate.py` puts eval/ at sys.path[0], so `evaluate`
        # resolves to our script and silently shadows the library (every score
        # came back 0.0). BERTScorer also caches roberta-large across calls.
        if _bertscore_metric is None:
            from bert_score import BERTScorer
            model_type = os.getenv('BERTSCORE_MODEL', 'roberta-large')
            _bertscore_metric = BERTScorer(model_type=model_type, lang='en',
                                           rescale_with_baseline=True, idf=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F = _bertscore_metric.score(predictions, references)
        rescaled_f1 = F.mean().item()
        # Rescaling is linear: r' = (r - b) / (1 - b) with fixed baseline b,
        # so the raw score is recovered exactly without a second model load.
        try:
            b = float(_bertscore_metric.baseline_vals[2].item())
            raw_f1 = rescaled_f1 * (1 - b) + b
        except Exception:
            raw_f1 = rescaled_f1
        return {
            'raw_f1':      round(raw_f1, 4),
            'rescaled_f1': round(rescaled_f1, 4),
            'bonus_hit':   rescaled_f1 >= 0.55 or raw_f1 >= 0.88,
        }
    except Exception as e:
        # Surface the failure loudly — a silent 0.0 once masked an entire run's scores.
        print(f'  [bertscore ERROR] {str(e)[:200]}', flush=True)
        return {'raw_f1': 0.0, 'rescaled_f1': 0.0, 'bonus_hit': False, 'error': str(e)}
