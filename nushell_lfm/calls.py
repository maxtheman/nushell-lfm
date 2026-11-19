"""The text of a run_nushell tool call, as the model writes it after the opening `<tool_call>` line."""

import re

_CALL = re.compile(r"\A<function=run_nushell>\n<parameter=code>\n(.*)\n</parameter>\n</function>\n</tool_call>\s*\Z", re.S)


def wrap_tool_call(code: str) -> str:
    return f"<function=run_nushell>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"


def parse_tool_call(completion: str):
    """The Nushell source of a well-formed run_nushell call, else None."""
    found = _CALL.match(completion)
    return found.group(1) if found else None
