"""NativeReActAgent -- Thought-Action-Observation loop agent.

Renamed from ``ReActAgent`` to clarify this is OpenJarvis's native
implementation, not an integration with an external project.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult, _message_to_dict
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool, build_tool_descriptions

REACT_SYSTEM_PROMPT = """\
You are a ReAct agent. For each step, respond with exactly one of:

1. To think and act:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <json arguments>

2. To give a final answer:
Thought: <your reasoning>
Final Answer: <your answer>

{tool_descriptions}"""


@AgentRegistry.register("native_react")
class NativeReActAgent(ToolUsingAgent):
    """ReAct agent: Thought -> Action -> Observation loop."""

    agent_id = "native_react"
    _default_temperature = 0.7
    _default_max_tokens = 1024
    _default_max_turns = 10

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        interactive: bool = False,
        confirm_callback=None,
        loop_guard_config: Optional[dict] = None,
    ) -> None:
        if loop_guard_config is None:
            loop_guard_config = {"enabled": True, "max_repeated_calls": 3, "max_turns_without_progress": 5}
        super().__init__(
            engine,
            model,
            tools=tools,
            bus=bus,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            interactive=interactive,
            confirm_callback=confirm_callback,
            loop_guard_config=loop_guard_config,
        )

    def _parse_response(self, text: str) -> dict:
        """Parse ReAct structured output."""
        result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(
            r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # Extract Action and Action Input
        action_match = re.search(r"Action:\s*(.+)", text, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\n\n|\nThought:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        # Build system prompt with rich tool descriptions
        tool_desc = build_tool_descriptions(self._tools)
        react_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

        # Prepend identity prompt from config (e.g. JARVIS persona)
        try:
            from openjarvis.core.config import load_config

            cfg = load_config()
            identity = cfg.agent.default_system_prompt
            if identity:
                system_prompt = f"{identity}\n\n{react_prompt}"
            else:
                system_prompt = react_prompt
        except Exception:
            system_prompt = react_prompt

        # Memory context injection
        try:
            from openjarvis.memory.jarvis_memory import search_memory, format_memory_context
            _mem_results = search_memory(input, top_k=3)
            _mem_context = format_memory_context(_mem_results)
            if _mem_context:
                input = f"{_mem_context}\n\n{input}"
        except Exception:
            pass

        messages = self._build_messages(input, context, system_prompt=system_prompt)

        all_tool_results: list[ToolResult] = []
        turns = 0
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _turn in range(self._max_turns):
            turns += 1

            if self._loop_guard:
                messages = self._loop_guard.compress_context(messages)

            result = self._generate(messages)
            usage = result.get("usage", {})
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            content = result.get("content", "")
            parsed = self._parse_response(content)

            # Final answer?
            if parsed["final_answer"]:
                self._emit_turn_end(turns=turns)
                msg_dicts = [_message_to_dict(m) for m in messages]
                try:
                    from openjarvis.memory.jarvis_memory import index_conversation
                    _raw_input = input.split("[FIN MEMORIA]\n\n", 1)[-1] if "[FIN MEMORIA]\n\n" in input else input
                    index_conversation(
                        user_input=_raw_input,
                        assistant_response=parsed["final_answer"],
                        agent=self.agent_id,
                        channel=getattr(context, "channel", "cli") if context else "cli",
                    )
                except Exception:
                    pass
                return AgentResult(
                    content=parsed["final_answer"],
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={**total_usage, "messages": msg_dicts},
                )

            # No action? Treat content as final answer
            if not parsed["action"]:
                self._emit_turn_end(turns=turns)
                msg_dicts = [_message_to_dict(m) for m in messages]
                try:
                    from openjarvis.memory.jarvis_memory import index_conversation
                    _raw_input = input.split("[FIN MEMORIA]\n\n", 1)[-1] if "[FIN MEMORIA]\n\n" in input else input
                    index_conversation(
                        user_input=_raw_input,
                        assistant_response=content,
                        agent=self.agent_id,
                        channel=getattr(context, "channel", "cli") if context else "cli",
                    )
                except Exception:
                    pass
                return AgentResult(
                    content=content,
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={**total_usage, "messages": msg_dicts},
                )

            # Execute action
            messages.append(Message(role=Role.ASSISTANT, content=content))

            tool_call = ToolCall(
                id=f"react_{turns}",
                name=parsed["action"],
                arguments=parsed["action_input"] or "{}",
            )

            # Loop guard check before execution
            if self._loop_guard:
                verdict = self._loop_guard.check_call(
                    tool_call.name,
                    tool_call.arguments,
                )
                if verdict.blocked:
                    tool_result = ToolResult(
                        tool_name=tool_call.name,
                        content=f"Loop guard: {verdict.reason}",
                        success=False,
                    )
                    all_tool_results.append(tool_result)
                    observation = f"Observation: {tool_result.content}"
                    messages.append(Message(role=Role.USER, content=observation))
                    continue

            tool_result = self._executor.execute(tool_call)
            all_tool_results.append(tool_result)

            observation = f"Observation: {tool_result.content}"
            messages.append(Message(role=Role.USER, content=observation))
            
            if self._loop_guard is not None:
                guard_result = self._loop_guard.check(messages, all_tool_results)
                if guard_result.triggered:
                    self._emit_turn_end(turns=turns, loop_guard_triggered=True)
                    return AgentResult(
                        content="He detectado un bucle en mi razonamiento. ¿Puedes reformular la pregunta, Pau?",
                        tool_results=all_tool_results,
                        turns=turns,
                        metadata={"loop_guard_triggered": True, "reason": guard_result.reason},
                    )

        # Max turns exceeded
        msg_dicts = [_message_to_dict(m) for m in messages]
        return self._max_turns_result(
            all_tool_results,
            turns,
            metadata={**total_usage, "messages": msg_dicts},
        )


__all__ = ["NativeReActAgent", "REACT_SYSTEM_PROMPT"]
