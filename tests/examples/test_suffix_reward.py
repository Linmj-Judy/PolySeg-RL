import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from examples.polyseg_trainer import suffix_reward
from examples.polyseg_trainer.suffix_reward import _jsonable, _normalise_prompt


def test_jsonable_converts_object_array_chat_prompt():
    prompt = np.asarray([[{"role": "user", "content": "Compute 2+2"}]], dtype=object)

    assert _jsonable(prompt) == [[{"role": "user", "content": "Compute 2+2"}]]
    assert _normalise_prompt(prompt) == [{"role": "user", "content": "Compute 2+2"}]


def test_compute_score_accepts_boxed_suffix_via_strict_verifier(monkeypatch):
    monkeypatch.setenv("POLYSEG_SUFFIX_SAMPLES", "1")
    monkeypatch.setattr(suffix_reward, "_sample_suffixes", lambda *args: [r"\boxed{72}"])

    result = suffix_reward.compute_score(
        "openai/gsm8k",
        "",
        "72",
        {"polyseg_protocol": "utility_delta", "raw_prompt": [], "budget_remaining": 1},
    )

    assert result["utility"] == 1.0


def test_suffix_reports_observed_final_correctness_separately(monkeypatch):
    monkeypatch.setenv("POLYSEG_SUFFIX_SAMPLES", "1")
    monkeypatch.setattr(suffix_reward, "_sample_suffixes", lambda *args: [r"\boxed{72}"])
    result = suffix_reward.compute_score(
        "openai/gsm8k",
        "",
        "72",
        {
            "polyseg_protocol": "utility_delta",
            "raw_prompt": [],
            "full_response": r"\boxed{72}",
            "segment_id": 0,
            "segment_count": 1,
            "budget_remaining": 1,
        },
    )
    assert result["utility"] == 1.0
    assert result["acc"] is True and result["verifier_score"] == 1.0


def test_suffix_endpoint_requires_exact_requested_cardinality(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"suffixes": ["a", "b"]}).encode()

    monkeypatch.setenv("POLYSEG_SUFFIX_ENDPOINT", "http://suffix.test")
    monkeypatch.setattr(suffix_reward, "urlopen", lambda *args, **kwargs: _Response())
    try:
        suffix_reward._sample_suffixes([], "", 1)
    except ValueError as exc:
        assert "expected exactly 1" in str(exc)
    else:
        raise AssertionError("over-returning suffix endpoint must be rejected")


def test_suffix_endpoint_preserves_provider_billing_receipt(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "suffixes": ["a"],
                    "billing_receipt": {
                        "provider_query_id": "provider-q-1",
                        "billed_cost": 1.0,
                    },
                }
            ).encode()

    monkeypatch.setenv("POLYSEG_SUFFIX_ENDPOINT", "http://suffix.test")
    monkeypatch.setattr(suffix_reward, "urlopen", lambda *args, **kwargs: _Response())
    suffixes = suffix_reward._sample_suffixes([], "", 1)
    assert suffixes == ["a"]
    assert suffixes.billing_receipt == {"provider_query_id": "provider-q-1", "billed_cost": 1.0}


def test_strict_suffix_score_accepts_structured_verifier(monkeypatch):
    monkeypatch.setattr(
        suffix_reward,
        "default_compute_score",
        lambda *args, **kwargs: {"score": 1.0, "acc": 1.0, "pred": "72"},
    )
    assert suffix_reward._strict_suffix_score("math_dapo", "answer", "72", {}) == 1.0


def test_suffix_compute_score_uses_real_http_multi_sample_receipt_transport(monkeypatch):
    requests = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            payload = {
                "suffixes": [r"\boxed{72}", r"\boxed{71}", r"\boxed{72}"],
                "billing_receipt": {
                    "provider_query_id": "local-provider-q-3",
                    "billed_cost": 1.5,
                },
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("POLYSEG_SUFFIX_ENDPOINT", f"http://127.0.0.1:{server.server_port}/suffix")
        monkeypatch.setenv("POLYSEG_SUFFIX_SAMPLES", "3")
        monkeypatch.setenv("POLYSEG_SUFFIX_COST_PER_SAMPLE", "0.5")
        monkeypatch.setenv("POLYSEG_SUFFIX_SEED", "17")
        result = suffix_reward.compute_score(
            "openai/gsm8k",
            "",
            "72",
            {
                "polyseg_protocol": "utility_delta",
                "raw_prompt": [[{"role": "user", "content": "Compute 72"}]],
                "segment_prefix": "",
                "budget_remaining": 1.5,
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert result["utility"] == 2 / 3
    assert result["query_cost"] == 1.5
    assert result["suffix_samples"] == 3
    assert result["billing_receipt"] == {
        "provider_query_id": "local-provider-q-3",
        "billed_cost": 1.5,
    }
    assert len(requests) == 1
    assert requests[0]["num_samples"] == 3
    assert requests[0]["seed"] == 17
    assert requests[0]["cost_per_sample"] == 0.5
    assert requests[0]["prompt"] == [{"role": "user", "content": "Compute 72"}]
