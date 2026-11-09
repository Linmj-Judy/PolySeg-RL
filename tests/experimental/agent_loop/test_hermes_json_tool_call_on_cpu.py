"""CPU contract tests for the opt-in Hermes JSON tool-call fallback."""

import asyncio

from verl.experimental.agent_loop.tool_parser import HermesToolParser


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, _ids):
        return self.text


def _parse(text: str, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("POLYSEG_GUIDED_TOOL_CALL", "true")
    parser = HermesToolParser(_Tokenizer(text))
    return asyncio.run(parser.extract_tool_calls([1]))


def test_json_tool_call_is_parsed_when_xml_wrapper_is_absent(monkeypatch):
    content, calls = _parse('{"name":"calculator","arguments":{"expression":"48+24"}}', monkeypatch)
    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].arguments == '{"expression": "48+24"}'


def test_malformed_json_remains_a_non_tool_response(monkeypatch):
    content, calls = _parse('{"name":"calculator","arguments":', monkeypatch)
    assert content.startswith('{"name"')
    assert calls == []


def test_existing_xml_path_is_unchanged():
    content, calls = _parse('<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>')
    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "calculator"


def test_plain_json_is_not_a_call_without_opt_in(monkeypatch):
    monkeypatch.delenv("POLYSEG_GUIDED_TOOL_CALL", raising=False)
    text = '{"name":"calculator","arguments":{"expression":"48+24"}}'
    content, calls = _parse(text)
    assert content == text
    assert calls == []
