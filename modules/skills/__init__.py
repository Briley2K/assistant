"""
Skill registry + tool-call protocol for the assistant.

A "skill" is a named tool the model can call: it has a description, a parameter
spec, and a handler that returns a JSON-serializable result. The model requests
a tool by emitting a single line:

    [[TOOL]] {"name": "<skill>", "args": { ... }}

llm.py detects that, runs the skill via dispatch(), feeds the result back, and
the model then answers conversationally. Adding a skill is ~20 lines — decorate
a function with @skill(...) in builtins.py.
"""
import json

TOOL_TOKEN = "[[TOOL]]"
MAX_TOOL_STEPS = 3            # safety cap on tool calls per turn

_REGISTRY: dict = {}


class Skill:
    def __init__(self, name, description, parameters, handler):
        self.name = name
        self.description = description
        self.parameters = parameters      # {arg_name: "description"}
        self.handler = handler


def skill(name, description, parameters=None):
    """Decorator registering a function as a callable skill."""
    def deco(fn):
        _REGISTRY[name] = Skill(name, description, parameters or {}, fn)
        return fn
    return deco


def all_skills() -> list:
    return list(_REGISTRY.values())


def is_enabled(name: str) -> bool:
    """A skill is active unless disabled in settings (config.SKILLS_DISABLED)."""
    import config
    return name not in config.SKILLS_DISABLED


def active_skills() -> list:
    return [s for s in _REGISTRY.values() if is_enabled(s.name)]


def dispatch(name: str, args: dict) -> dict:
    """Run a skill by name. Always returns a JSON-serializable dict."""
    skill_obj = _REGISTRY.get(name)
    if skill_obj is None:
        return {"error": f"unknown tool '{name}'"}
    if not is_enabled(name):
        return {"error": f"tool '{name}' is disabled"}
    try:
        result = skill_obj.handler(args or {})
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tools_prompt() -> str:
    """Build the system-prompt section describing available tools + protocol."""
    skills = active_skills()
    if not skills:
        return ""
    lines = [
        "You have access to tools. When a tool is needed to answer (current time,",
        "weather, setting a timer, etc.), respond with ONLY this single line and",
        "nothing else:",
        '  [[TOOL]] {"name": "<tool>", "args": {<arguments>}}',
        "Do not describe the call or add other text. After the tool result comes",
        "back, answer the user naturally and briefly. If no tool is needed, just",
        "answer normally. Available tools:",
    ]
    for s in skills:
        if s.parameters:
            params = ", ".join(f"{k} ({v})" for k, v in s.parameters.items())
        else:
            params = "no arguments"
        lines.append(f"- {s.name}: {s.description} Args: {params}")
    return "\n".join(lines)


def _extract_json(text: str) -> str | None:
    """Return the first balanced {...} object found in text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_tool_call(text: str):
    """If `text` is a tool call, return (name, args); else None. Tolerates
    markdown fences and stray text around the JSON."""
    if TOOL_TOKEN not in text:
        return None
    after = text.split(TOOL_TOKEN, 1)[1]
    blob = _extract_json(after)
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("tool")
    args = data.get("args") or data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    if name in _REGISTRY:
        return name, args
    return None


# Register the built-in skills on import.
from modules.skills import builtins as _builtins  # noqa: E402,F401
from modules.skills import apps as _apps          # noqa: E402,F401
