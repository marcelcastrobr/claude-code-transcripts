"""Tests for Snowflake Cortex Code CLI support."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_code_transcripts import (
    cli,
    is_cortex_session_file,
    normalize_cortex_content_block,
    normalize_cortex_content,
    parse_cortex_session_file,
    get_cortex_session_summary,
    find_cortex_sessions,
    parse_session_file,
    generate_html,
)


@pytest.fixture
def sample_cortex_session():
    """Load the sample Cortex session fixture."""
    fixture_path = Path(__file__).parent / "sample_cortex_session.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def cortex_session_file():
    """Create a temporary Cortex session file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {
            "title": "Test Cortex Session",
            "session_id": "test-session-id",
            "working_directory": "/test/path",
            "history": [
                {
                    "role": "user",
                    "id": "user1",
                    "user_sent_time": "2026-01-05T10:00:00.000000",
                    "content": [{"type": "text", "text": "Hello Cortex"}],
                },
                {
                    "role": "assistant",
                    "id": "asst1",
                    "content": [{"type": "text", "text": "Hello! How can I help?"}],
                },
            ],
        }
        json.dump(data, f)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestIsCortexSessionFile:
    """Tests for is_cortex_session_file function."""

    def test_detects_cortex_json_file(self, cortex_session_file):
        """Test that Cortex JSON files are correctly detected."""
        assert is_cortex_session_file(cortex_session_file) is True

    def test_rejects_claude_code_json(self, tmp_path):
        """Test that Claude Code JSON files are not detected as Cortex."""
        claude_file = tmp_path / "claude.json"
        claude_file.write_text(json.dumps({"loglines": []}))
        assert is_cortex_session_file(claude_file) is False

    def test_rejects_jsonl_files(self, tmp_path):
        """Test that JSONL files are not detected as Cortex."""
        jsonl_file = tmp_path / "session.jsonl"
        jsonl_file.write_text('{"type": "user"}\n')
        assert is_cortex_session_file(jsonl_file) is False

    def test_rejects_nonexistent_file(self):
        """Test that non-existent files return False."""
        assert is_cortex_session_file(Path("/nonexistent/file.json")) is False


class TestNormalizeCortexContentBlock:
    """Tests for normalize_cortex_content_block function."""

    def test_normalizes_thinking_block(self):
        """Test normalization of nested thinking blocks."""
        cortex_block = {
            "type": "thinking",
            "thinking": {"text": "Thinking text here", "signature": "sig123"},
        }
        result = normalize_cortex_content_block(cortex_block)
        assert result == {"type": "thinking", "thinking": "Thinking text here"}

    def test_preserves_simple_thinking_block(self):
        """Test that simple thinking blocks pass through unchanged."""
        simple_block = {"type": "thinking", "thinking": "Simple text"}
        result = normalize_cortex_content_block(simple_block)
        assert result == simple_block

    def test_normalizes_tool_use_block(self):
        """Test normalization of nested tool_use blocks."""
        cortex_block = {
            "type": "tool_use",
            "tool_use": {
                "tool_use_id": "toolu_123",
                "name": "Read",
                "input": {"file_path": "/test.py"},
            },
        }
        result = normalize_cortex_content_block(cortex_block)
        assert result == {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "Read",
            "input": {"file_path": "/test.py"},
        }

    def test_normalizes_tool_result_block(self):
        """Test normalization of nested tool_result blocks."""
        cortex_block = {
            "type": "tool_result",
            "tool_result": {
                "tool_use_id": "toolu_123",
                "content": [{"type": "text", "text": "File contents here"}],
                "status": "success",
            },
        }
        result = normalize_cortex_content_block(cortex_block)
        assert result == {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "File contents here",
        }

    def test_passes_through_text_blocks(self):
        """Test that text blocks pass through unchanged."""
        text_block = {"type": "text", "text": "Some text"}
        result = normalize_cortex_content_block(text_block)
        assert result == text_block


class TestNormalizeCortexContent:
    """Tests for normalize_cortex_content function."""

    def test_normalizes_list_of_blocks(self):
        """Test normalizing a list of content blocks."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": {"text": "Thinking", "signature": "sig"}},
        ]
        result = normalize_cortex_content(content)
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "Thinking"},
        ]

    def test_handles_non_list_input(self):
        """Test that non-list input is returned as-is."""
        result = normalize_cortex_content("string content")
        assert result == "string content"


class TestParseCortexSessionFile:
    """Tests for parse_cortex_session_file function."""

    def test_parses_cortex_format(self, cortex_session_file):
        """Test parsing a Cortex session file."""
        result = parse_cortex_session_file(cortex_session_file)

        assert "loglines" in result
        assert len(result["loglines"]) == 2

        # Check first entry (user)
        user_entry = result["loglines"][0]
        assert user_entry["type"] == "user"
        assert user_entry["timestamp"] == "2026-01-05T10:00:00.000000"
        assert user_entry["message"]["role"] == "user"

        # Check second entry (assistant)
        asst_entry = result["loglines"][1]
        assert asst_entry["type"] == "assistant"

    def test_filters_system_reminders(self, tmp_path):
        """Test that system-reminder blocks are filtered from user messages."""
        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps(
                {
                    "title": "Test",
                    "session_id": "test",
                    "history": [
                        {
                            "role": "user",
                            "id": "user1",
                            "user_sent_time": "2026-01-05T10:00:00",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<system-reminder>Ignore this</system-reminder>What is Python?",
                                }
                            ],
                        }
                    ],
                }
            )
        )

        result = parse_cortex_session_file(session_file)
        user_content = result["loglines"][0]["message"]["content"]
        # The system-reminder should be filtered out
        assert len(user_content) == 1
        assert "system-reminder" not in user_content[0]["text"]
        assert "What is Python?" in user_content[0]["text"]


class TestGetCortexSessionSummary:
    """Tests for get_cortex_session_summary function."""

    def test_returns_title_if_available(self, tmp_path):
        """Test that session title is used as summary."""
        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps(
                {
                    "title": "My Test Session Title",
                    "session_id": "test",
                    "history": [],
                }
            )
        )

        summary = get_cortex_session_summary(session_file)
        assert summary == "My Test Session Title"

    def test_falls_back_to_first_user_message(self, tmp_path):
        """Test fallback to first user message when no title."""
        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps(
                {
                    "title": "",
                    "session_id": "test",
                    "history": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Hello world"}],
                        }
                    ],
                }
            )
        )

        summary = get_cortex_session_summary(session_file)
        assert summary == "Hello world"

    def test_truncates_long_summary(self, tmp_path):
        """Test that long summaries are truncated."""
        session_file = tmp_path / "session.json"
        long_title = "A" * 300
        session_file.write_text(
            json.dumps({"title": long_title, "session_id": "test", "history": []})
        )

        summary = get_cortex_session_summary(session_file, max_length=50)
        assert len(summary) == 50
        assert summary.endswith("...")

    def test_returns_no_summary_for_invalid_file(self):
        """Test that invalid files return no summary."""
        summary = get_cortex_session_summary(Path("/nonexistent.json"))
        assert summary == "(no summary)"


class TestFindCortexSessions:
    """Tests for find_cortex_sessions function."""

    def test_finds_sessions_in_directory(self, tmp_path):
        """Test finding Cortex sessions in a directory."""
        # Create two session files
        for i in range(2):
            session_file = tmp_path / f"session{i}.json"
            session_file.write_text(
                json.dumps(
                    {
                        "title": f"Session {i}",
                        "session_id": f"session-{i}",
                        "history": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Hello"}],
                            }
                        ],
                    }
                )
            )

        results = find_cortex_sessions(tmp_path, limit=10)
        assert len(results) == 2

    def test_respects_limit(self, tmp_path):
        """Test that limit parameter is respected."""
        # Create 5 session files
        for i in range(5):
            session_file = tmp_path / f"session{i}.json"
            session_file.write_text(
                json.dumps(
                    {
                        "title": f"Session {i}",
                        "session_id": f"session-{i}",
                        "history": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Hello"}],
                            }
                        ],
                    }
                )
            )

        results = find_cortex_sessions(tmp_path, limit=2)
        assert len(results) == 2

    def test_skips_non_cortex_files(self, tmp_path):
        """Test that non-Cortex JSON files are skipped."""
        # Create a Cortex session
        cortex_file = tmp_path / "cortex.json"
        cortex_file.write_text(
            json.dumps(
                {
                    "title": "Cortex Session",
                    "session_id": "cortex-1",
                    "history": [
                        {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
                    ],
                }
            )
        )

        # Create a non-Cortex JSON file
        other_file = tmp_path / "other.json"
        other_file.write_text(json.dumps({"foo": "bar"}))

        results = find_cortex_sessions(tmp_path)
        assert len(results) == 1
        assert results[0][1] == "Cortex Session"

    def test_returns_empty_for_nonexistent_directory(self):
        """Test that non-existent directories return empty list."""
        results = find_cortex_sessions(Path("/nonexistent/path"))
        assert results == []


class TestParseSessionFileWithCortex:
    """Tests for parse_session_file with Cortex files."""

    def test_auto_detects_cortex_format(self, cortex_session_file):
        """Test that parse_session_file auto-detects Cortex format."""
        result = parse_session_file(cortex_session_file)

        assert "loglines" in result
        assert len(result["loglines"]) == 2
        assert result["loglines"][0]["type"] == "user"

    def test_still_handles_jsonl(self, tmp_path):
        """Test that JSONL files still work."""
        jsonl_file = tmp_path / "session.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "timestamp": "2026-01-01T10:00:00.000Z", "message": {"role": "user", "content": "Hello"}}\n'
        )

        result = parse_session_file(jsonl_file)
        assert "loglines" in result


class TestGenerateHtmlWithCortex:
    """Tests for generate_html with Cortex session files."""

    def test_generates_html_from_cortex_file(self, output_dir):
        """Test HTML generation from Cortex session file."""
        fixture_path = Path(__file__).parent / "sample_cortex_session.json"
        generate_html(fixture_path, output_dir)

        # Verify HTML files were created
        assert (output_dir / "index.html").exists()
        assert (output_dir / "page-001.html").exists()

        # Check index contains expected content
        index_html = (output_dir / "index.html").read_text()
        # The title or user question should appear
        assert "portfolio_role" in index_html or "connection" in index_html.lower()

    def test_normalizes_cortex_tool_blocks(self, output_dir, tmp_path):
        """Test that Cortex tool blocks are properly normalized for rendering."""
        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps(
                {
                    "title": "Tool Test",
                    "session_id": "test",
                    "history": [
                        {
                            "role": "user",
                            "id": "u1",
                            "user_sent_time": "2026-01-05T10:00:00",
                            "content": [{"type": "text", "text": "Run a command"}],
                        },
                        {
                            "role": "assistant",
                            "id": "a1",
                            "content": [
                                {"type": "text", "text": "Running command..."},
                                {
                                    "type": "tool_use",
                                    "tool_use": {
                                        "tool_use_id": "toolu_123",
                                        "name": "Bash",
                                        "input": {"command": "echo hello"},
                                    },
                                },
                            ],
                        },
                    ],
                }
            )
        )

        generate_html(session_file, output_dir)

        page_html = (output_dir / "page-001.html").read_text()
        # The tool name should appear in the rendered output
        assert "Bash" in page_html


class TestCortexCommand:
    """Tests for the cortex CLI command."""

    def test_cortex_command_exists(self):
        """Test that cortex command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cortex", "--help"])
        assert result.exit_code == 0
        assert "Cortex" in result.output or "cortex" in result.output

    def test_cortex_command_handles_missing_directory(self, tmp_path):
        """Test cortex command with non-existent directory."""
        runner = CliRunner()
        nonexistent = tmp_path / "nonexistent"
        result = runner.invoke(cli, ["cortex", "--source", str(nonexistent)])
        # Should fail because source path doesn't exist
        assert result.exit_code != 0

    def test_cortex_command_with_custom_source(self, tmp_path, output_dir):
        """Test cortex command with custom source directory."""
        # Create a session file
        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps(
                {
                    "title": "Test Session",
                    "session_id": "test",
                    "history": [
                        {
                            "role": "user",
                            "id": "u1",
                            "user_sent_time": "2026-01-05T10:00:00",
                            "content": [{"type": "text", "text": "Hello"}],
                        },
                        {
                            "role": "assistant",
                            "id": "a1",
                            "content": [{"type": "text", "text": "Hi there!"}],
                        },
                    ],
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["cortex", "--source", str(tmp_path)])

        # Should find the session
        assert "Loading Cortex" in result.output or "Test Session" in result.output
