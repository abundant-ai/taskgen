from __future__ import annotations

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


# ANSI color codes for verbose output
class Colors:
    BLUE = "\033[94m"  # Assistant messages
    CYAN = "\033[96m"  # Tool use
    MAGENTA = "\033[95m"  # Tool results
    GREEN = "\033[92m"  # Success/final result
    YELLOW = "\033[93m"  # System messages
    RED = "\033[91m"  # Errors
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_sdk_message(message: object) -> None:
    """Print SDK messages with colored formatting.

    Message types include:
    - AssistantMessage: Claude's text responses and tool uses
    - UserMessage: User messages (for tool results in SDK)
    - ResultMessage: Final result message
    - SystemMessage: System notifications
    """
    if isinstance(message, AssistantMessage):
        # Assistant message content
        for block in message.content:
            if isinstance(block, TextBlock):
                text = block.text
                if text.strip():
                    print(f"\n{Colors.BLUE}[Assistant]{Colors.RESET} {text}", flush=True)
            elif isinstance(block, ToolUseBlock):
                tool_name = block.name.upper()
                tool_input = block.input
                # Less aggressive truncation - commands are important!
                summary: dict | str
                if isinstance(tool_input, dict):
                    # For bash commands, show up to 2000 chars; for other inputs, 1000 chars
                    max_len = 2000 if tool_name.lower() == "bash" else 1000
                    summary = {
                        k: (v[:max_len] + "..." if isinstance(v, str) and len(v) > max_len else v)
                        for k, v in tool_input.items()
                    }
                else:
                    summary = str(tool_input)[:2000]
                print(
                    f"\n{Colors.CYAN}{Colors.BOLD}{tool_name}{Colors.RESET}: {summary}",
                    flush=True,
                )

    elif isinstance(message, UserMessage):
        # In SDK mode, tool results come as UserMessage with ToolResultBlock
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                content = block.content if hasattr(block, "content") else str(block)
                # Less aggressive truncation - show up to 2000 chars
                if isinstance(content, str) and len(content) > 2000:
                    content = content[:2000] + f"... ({len(content)} chars total)"
                print(f"{Colors.MAGENTA}[Tool Result]{Colors.RESET} {content}", flush=True)
            elif isinstance(block, TextBlock):
                text = block.text
                if text.strip():
                    print(f"{Colors.MAGENTA}[Tool Result]{Colors.RESET} {text}", flush=True)

    elif isinstance(message, ResultMessage):
        # Final result message
        result_text = getattr(message, "text", str(message))
        if result_text and result_text.strip():
            if len(result_text) > 3000:
                result_text = result_text[:3000] + f"... ({len(result_text)} chars total)"
            print(
                f"\n{Colors.GREEN}{Colors.BOLD}[Final Result]{Colors.RESET}\n{result_text}",
                flush=True,
            )

    elif isinstance(message, SystemMessage):
        # System messages
        msg_text = getattr(message, "text", str(message))
        if msg_text:
            print(f"{Colors.YELLOW}[System]{Colors.RESET} {msg_text}", flush=True)
