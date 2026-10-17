"""
Tests for `hermes mcp call` (cmd_mcp_call) — one-shot MCP tool invocation.

Connection layer is mocked (same convention as test_mcp_config.py), so no real
MCP server or key is needed.

Exit-code contract under test:
  0 = tool answered, 1 = connection/tool error, 3 = server not in config,
  4 = unparseable --args (argparse keeps 2 for usage errors).
"""

import argparse

import pytest


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect all config I/O to a temp directory (mirrors test_mcp_config.py)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: tmp_path / ".env")
    return tmp_path


def _args(**kwargs):
    defaults = {"name": "ink", "tool": "echo", "tool_args": "{}", "mcp_action": "call"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _seed_config(tmp_path, mcp_servers: dict) -> None:
    import hermes_yaml as yaml

    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"mcp_servers": mcp_servers, "_config_version": 9}, f)


class TestMcpCall:
    def test_success_prints_result_and_exits_0(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {"ink": {"url": "https://mcp.ml.ink/mcp"}})
        from hermes_cli.mcp_config import cmd_mcp_call

        captured = {}

        def fake_once(name, cfg, tool, tool_args, timeout):
            captured.update(name=name, tool=tool, tool_args=tool_args, timeout=timeout)
            return False, "the answer is 42"

        monkeypatch.setattr("hermes_cli.mcp_config._call_tool_once", fake_once)
        assert cmd_mcp_call(_args()) == 0
        assert captured["name"] == "ink"
        assert captured["tool"] == "echo"
        assert captured["tool_args"] == {}
        assert captured["timeout"] > 0
        assert "the answer is 42" in capsys.readouterr().out

    def test_arguments_json_is_parsed_into_a_dict(self, tmp_path, monkeypatch):
        _seed_config(tmp_path, {"ink": {"url": "https://mcp.ml.ink/mcp"}})
        from hermes_cli.mcp_config import cmd_mcp_call

        captured = {}

        def fake_once(name, cfg, tool, tool_args, timeout):
            captured["tool_args"] = tool_args
            return False, "ok"

        monkeypatch.setattr("hermes_cli.mcp_config._call_tool_once", fake_once)
        assert cmd_mcp_call(_args(tool_args='{"n": 10, "level": "ERROR"}')) == 0
        assert captured["tool_args"] == {"n": 10, "level": "ERROR"}

    def test_exit_codes_distinguish_every_failure(self, tmp_path, capsys, monkeypatch):
        """3 unknown server, 4 bad --args, 1 connection/tool error — never a silent 0."""
        _seed_config(tmp_path, {"ink": {"url": "https://mcp.ml.ink/mcp"}})
        from hermes_cli.mcp_config import cmd_mcp_call

        assert cmd_mcp_call(_args(name="doesnotexist")) == 3
        assert "not found in config" in capsys.readouterr().out

        assert cmd_mcp_call(_args(tool_args="{broken")) == 4
        assert "JSON" in capsys.readouterr().out

        def failing_once(name, cfg, tool, tool_args, timeout):
            raise RuntimeError("Server returned an error response")

        monkeypatch.setattr("hermes_cli.mcp_config._call_tool_once", failing_once)
        assert cmd_mcp_call(_args()) == 1

    def test_tool_level_error_exits_1_and_prints_payload(self, tmp_path, capsys, monkeypatch):
        """isError=true from the server: exit 1, the tool's own text goes to the output."""
        _seed_config(tmp_path, {"ink": {"url": "https://mcp.ml.ink/mcp"}})
        from hermes_cli.mcp_config import cmd_mcp_call

        def error_once(name, cfg, tool, tool_args, timeout):
            return True, "n must be between 1 and 10000"

        monkeypatch.setattr("hermes_cli.mcp_config._call_tool_once", error_once)
        assert cmd_mcp_call(_args()) == 1
        out = capsys.readouterr().out
        assert "n must be between 1 and 10000" in out

    def test_non_object_args_rejected(self, tmp_path, capsys, monkeypatch):
        """--args '[1,2]' is a list, not an object — rejected before any connection."""
        _seed_config(tmp_path, {"ink": {"url": "https://mcp.ml.ink/mcp"}})
        from hermes_cli.mcp_config import cmd_mcp_call

        def fail_if_called(*a, **kw):
            raise AssertionError("must not connect when --args is invalid")

        monkeypatch.setattr("hermes_cli.mcp_config._call_tool_once", fail_if_called)
        assert cmd_mcp_call(_args(tool_args="[1, 2]")) == 4
        assert "JSON" in capsys.readouterr().out

    def test_render_call_result_prefers_text_then_structured(self):
        """Content blocks join with newlines; structured JSON only when no text blocks.

        Field names follow the installed mcp SDK: snake_case (structured_content).
        """
        from hermes_cli.mcp_config import _render_call_result

        class Block:
            def __init__(self, text):
                self.text = text

        class Result:
            def __init__(self, content, structured_content=None):
                self.content = content
                self.structured_content = structured_content

        assert _render_call_result(Result([Block("line1"), Block("line2")])) == "line1\nline2"
        assert _render_call_result(Result([], structured_content={"a": 1})) == '{"a": 1}'

    def test_call_is_error_reads_snake_and_camel_field(self):
        """SDK models use is_error; tolerate isError too — never default a real error to success."""
        from hermes_cli.mcp_config import _call_is_error

        class Snake:
            is_error = True

        class Camel:
            isError = True

        class Clean:
            is_error = False

        class Bare:
            pass

        assert _call_is_error(Snake()) is True
        assert _call_is_error(Camel()) is True
        assert _call_is_error(Clean()) is False
        assert _call_is_error(Bare()) is False

    def test_cli_dispatcher_forwards_call_exit_code(self, tmp_path, monkeypatch):
        """`hermes mcp call` reaches the dispatcher with the handler's code."""
        _seed_config(tmp_path, {"ink": {"url": "https://mcp.ml.ink/mcp"}})
        from hermes_cli.mcp_config import mcp_command

        def failing_once(name, cfg, tool, tool_args, timeout):
            raise RuntimeError("boom")

        monkeypatch.setattr("hermes_cli.mcp_config._call_tool_once", failing_once)
        assert mcp_command(_args()) == 1
        assert mcp_command(_args(name="doesnotexist")) == 3
