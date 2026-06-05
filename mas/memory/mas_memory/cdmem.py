from dataclasses import dataclass
import copy
import os
import re
import uuid
from typing import Any, Optional

from langchain.docstore.document import Document
from langchain_chroma import Chroma

from .memory_base import MASMemoryBase
from ..common import MASMessage
from mas.llm import Message
from mas.utils import load_json, write_json


class CDMemPrompts:
    expert_system_prompt = (
        "You are an expert ALFWorld memory encoder. You compress a task "
        "trajectory into reusable environment observations and action chunks."
    )

    expert_user_prompt = """
Given an ALFWorld task and its action-observation trajectory, encode the useful memory.
Ignore private think actions when writing Expert Actions. Preserve exact object and receptacle names when they matter.

Return exactly this format:
Expert Observations:
(1) Locations: summarize useful object locations, or None.
(2) Functions: summarize useful receptacle/tool functions such as sinkbasin cleans, microwave heats, fridge cools, desklamp examines, or None.
Expert Actions:
summarize the executed action sequence in the original order.

Task:
{task}

Trajectory:
{trajectory}
""".strip()

    reflection_system_prompt = (
        "You are an ALFWorld reflection agent. You produce compact lessons "
        "that help future trials succeed and avoid repeated mistakes."
    )

    reflection_user_prompt = """
Task:
{task}

Status: {status}

Trajectory:
{trajectory}

Expert encoding:
{expert_result}

Past reflections for this same environment/task:
{local_reflections}

Write one concise reflection. If the task succeeded, identify the critical reusable steps.
If it failed, identify the failure type (planning, search, or operation) and the next correction.
Start with "Reflection:".
""".strip()

    env_summary_system_prompt = (
        "You update compact ALFWorld environment memory. Merge new function "
        "observations with the existing summary without duplicating facts."
    )

    env_summary_user_prompt = """
Existing environment summary:
{known_obs}

New function observations:
{increment_known_obs}

Return a concise updated summary of container/tool functions. Use numbered items.
""".strip()

    task_summary_system_prompt = (
        "You update compact ALFWorld task guidance. Merge new reflections "
        "with the existing guidance for one task type."
    )

    task_summary_user_prompt = """
Task type: {task_type}
Status bucket: {status}

Existing task guidance:
{action_guidance}

New experiences:
{increment_action_guidance}

Return concise reusable guidance. Use numbered items. Include both what to do and, for failures, what to avoid.
""".strip()


class CDMemShortMemory:
    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []

    def add(self, label: str, value: str) -> None:
        if label not in {"action", "observation"}:
            raise ValueError(f"Unsupported short-memory label: {label}")
        self.history.append({"label": label, "value": value})

    def reset(self) -> None:
        self.history = []

    def recall(self, with_think: bool = True) -> str:
        text = "\n"
        for i, item in enumerate(self.history):
            if item["label"] == "action":
                if not with_think and item["value"].startswith("think:"):
                    continue
                text += f'> {item["value"]}'
            else:
                text += item["value"]
            if i != len(self.history) - 1:
                text += "\n"
        return text


@dataclass
class CDMemMASMemory(MASMemoryBase):
    """
    CDMem-style memory adapted to the MASMemoryBase interface.

    The structure follows CDMem's short/local/global memory split:
    - short memory: current action-observation trajectory;
    - local memory: reflections keyed by ALFWorld game/task identity;
    - global memory: environment summaries and task-type guidance across trials.
    """

    def __post_init__(self):
        super().__post_init__()

        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=os.path.join(self.persist_dir, "trajectory_store"),
        )

        self.local_memory_path = os.path.join(self.persist_dir, "local_memory.json")
        self.env_memory_path = os.path.join(self.persist_dir, "global_env_memory.json")
        self.task_memory_path = os.path.join(self.persist_dir, "global_task_memory.json")
        self.trajectory_store_path = os.path.join(self.persist_dir, "expert_trajectories.json")

        self.local_memory: dict[str, dict[str, Any]] = self._load_dict(self.local_memory_path)
        self.env_memory: dict[str, dict[str, Any]] = self._load_dict(self.env_memory_path)
        self.task_memory: dict[str, dict[str, Any]] = self._load_dict(self.task_memory_path)
        self.trajectory_store: dict[str, dict[str, Any]] = self._load_dict(self.trajectory_store_path)

        self.short_memory = CDMemShortMemory()
        self.last_recall: dict[str, Any] = {}
        self.recalled_guidance_cache: list[str] = []

        self.env_batch_size = int(self.global_config.get("env_batch_size", 1))
        self.task_batch_size = int(self.global_config.get("task_batch_size", 1))
        self.local_reflection_limit = int(self.global_config.get("local_reflection_limit", 3))

        self._pending_task_key: Optional[str] = None
        self._pending_env_name: Optional[str] = None
        self.current_task_key: Optional[str] = None
        self.current_env_name: Optional[str] = None
        self.current_task_type: Optional[str] = None

        print(self._get_hyperparams_dict())

    def _get_hyperparams_dict(self) -> dict[str, Any]:
        return {
            "memory": "cdmem",
            "env_batch_size": self.env_batch_size,
            "task_batch_size": self.task_batch_size,
            "local_reflection_limit": self.local_reflection_limit,
            "working_dir": self.persist_dir,
        }

    @staticmethod
    def _load_dict(path: str) -> dict[str, Any]:
        data = load_json(path)
        return data if isinstance(data, dict) else {}

    def prepare_task(self, task_config: dict[str, Any]) -> None:
        """Set a stable local-memory key before init_task_context is called."""
        env_kwargs = task_config.get("env_kwargs", {})
        self._pending_task_key = (
            env_kwargs.get("gamefile")
            or task_config.get("task")
            or task_config.get("task_main")
        )
        self._pending_env_name = task_config.get("env_name")

    def init_task_context(self, task_main: str, task_description: str = None) -> MASMessage:
        context = super().init_task_context(task_main, task_description)
        self.short_memory.reset()
        self.current_task_key = self._pending_task_key or task_main
        self.current_env_name = self._pending_env_name
        _, task_text = self._split_alfworld_description(task_description or task_main)
        self.current_task_type = self._convert_task_description(task_text)
        self._ensure_local_slot(self.current_task_key)
        return context

    def move_memory_state(self, action: str, observation: str, **kargs) -> None:
        self.short_memory.add("action", action)
        self.short_memory.add("observation", "Observation: " + observation)
        super().move_memory_state(action, observation, **kargs)

    def add_memory(self, mas_message: MASMessage) -> None:
        if mas_message.label not in {True, False}:
            raise ValueError("The mas_message must have label!")

        mas_message_copy = copy.deepcopy(mas_message)
        expert_trajectory = self._build_expert_trajectory(mas_message_copy)
        mas_message_copy.add_extra_field("cdmem_expert", expert_trajectory)
        mas_message_copy.add_extra_field("task_type", expert_trajectory["task_type"])
        mas_message_copy.add_extra_field("local_memory_key", expert_trajectory["local_key"])

        self._update_local_memory(expert_trajectory)
        self._update_global_memory(expert_trajectory)

        memory_doc = Document(
            page_content=self._document_text(mas_message_copy, expert_trajectory),
            metadata=MASMessage.to_dict(mas_message_copy),
        )
        self.main_memory.add_documents([memory_doc])
        self._persist()
        self._index_done()

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        insight_topk: int = 3,
        threshold: float = 0.0,
        **kwargs,
    ) -> tuple[list[MASMessage], list[MASMessage], list[str]]:
        task_description = kwargs.get("task_description")
        if task_description is None and getattr(self, "current_task_context", None) is not None:
            task_description = self.current_task_context.task_description

        self.last_recall = self.recall_cdmem(task_description or query_task)
        successful = self._search_messages(query_task, True, successful_topk, threshold)
        failed = self._search_messages(query_task, False, failed_topk, threshold)

        guidance = self._recall_to_guidance(self.last_recall)
        if insight_topk is not None and insight_topk >= 0:
            guidance = guidance[:insight_topk]
        self.recalled_guidance_cache = guidance

        return successful, failed, guidance

    def recall_cdmem(self, task_description: str) -> dict[str, Any]:
        env_description, task_text = self._split_alfworld_description(task_description)
        task_type = self._convert_task_description(task_text)

        known_obs = ""
        if env_description in self.env_memory:
            known_obs = self.env_memory[env_description].get("known_obs", "")

        action_guidance = self._recall_task_guidance(task_type)
        local_reflections = []
        if self.current_task_key and self.current_task_key in self.local_memory:
            local_reflections = self.local_memory[self.current_task_key].get("reflection", [])
            local_reflections = local_reflections[-self.local_reflection_limit :]

        return {
            "env_description": env_description,
            "task_description": task_text,
            "task_type": task_type,
            "known_obs": known_obs,
            "action_guidance": action_guidance,
            "local_reflections": local_reflections,
        }

    def get_last_recall(self) -> dict[str, Any]:
        return self.last_recall

    def summarize(self, **kwargs) -> str:
        if getattr(self, "current_task_context", None) is None:
            return ""
        return self.current_task_context.task_description + self.short_memory.recall()

    def backward(self, reward, **kwargs) -> None:
        self.recalled_guidance_cache = []

    @property
    def memory_size(self) -> int:
        try:
            return len(self.main_memory.get()["ids"])
        except Exception:
            return 0

    def _build_expert_trajectory(self, mas_message: MASMessage) -> dict[str, Any]:
        env_description, task_text = self._split_alfworld_description(mas_message.task_description or "")
        task_type = self._convert_task_description(task_text)
        trajectory = mas_message.task_trajectory or ""

        expert_result = self._call_llm(
            CDMemPrompts.expert_system_prompt,
            CDMemPrompts.expert_user_prompt.format(
                task=mas_message.task_description,
                trajectory=trajectory,
            ),
            max_tokens=512,
            temperature=0.1,
        )
        location, function, action = self._parse_expert_result(expert_result)
        if not action:
            action = self._strip_thoughts(trajectory)

        local_reflections = self._get_local_reflections()
        reflection_result = self._call_llm(
            CDMemPrompts.reflection_system_prompt,
            CDMemPrompts.reflection_user_prompt.format(
                task=mas_message.task_description,
                status="success" if mas_message.label else "failure",
                trajectory=trajectory,
                expert_result=expert_result or action,
                local_reflections=self._format_list(local_reflections) or "None",
            ),
            max_tokens=256,
            temperature=0.1,
        )
        reflection = self._parse_reflection(reflection_result)
        if not reflection:
            reflection = self._fallback_reflection(mas_message.label, action)

        trajectory_id = uuid.uuid4().hex
        return {
            "id": trajectory_id,
            "local_key": self.current_task_key or mas_message.task_main,
            "task_main": mas_message.task_main,
            "env": env_description,
            "task": task_text,
            "task_type": task_type,
            "location": location or "None",
            "function": function or "None",
            "action": action or "None",
            "reflection": reflection,
            "is_success": bool(mas_message.label),
            "trajectory": trajectory,
        }

    def _update_local_memory(self, expert_trajectory: dict[str, Any]) -> None:
        key = expert_trajectory["local_key"]
        slot = self._ensure_local_slot(key)
        slot["env"] = expert_trajectory["env"]
        slot["task"] = expert_trajectory["task"]
        slot["task_type"] = expert_trajectory["task_type"]
        slot["location"] = expert_trajectory["location"]
        slot["function"] = expert_trajectory["function"]
        slot["action"] = expert_trajectory["action"]
        slot["is_success"] = bool(slot.get("is_success")) or expert_trajectory["is_success"]
        if expert_trajectory["reflection"]:
            slot.setdefault("reflection", []).append(expert_trajectory["reflection"])
            slot["reflection"] = slot["reflection"][-self.local_reflection_limit :]

    def _update_global_memory(self, expert_trajectory: dict[str, Any]) -> None:
        self.trajectory_store[expert_trajectory["id"]] = expert_trajectory
        increment_env, increment_task = self._short2long(expert_trajectory)

        if increment_env:
            summary = self._call_llm(
                CDMemPrompts.env_summary_system_prompt,
                CDMemPrompts.env_summary_user_prompt.format(
                    known_obs=increment_env.get("known_obs") or "None",
                    increment_known_obs=self._format_list(increment_env.get("increment_known_obs", [])),
                ),
                max_tokens=256,
                temperature=0.1,
            )
            env_key = expert_trajectory["env"]
            self.env_memory[env_key]["known_obs"] = summary or self._format_list(
                increment_env.get("increment_known_obs", [])
            )

        if increment_task:
            status = "success" if expert_trajectory["is_success"] else "fail"
            task_type = expert_trajectory["task_type"]
            summary = self._call_llm(
                CDMemPrompts.task_summary_system_prompt,
                CDMemPrompts.task_summary_user_prompt.format(
                    task_type=task_type,
                    status=status,
                    action_guidance=increment_task.get("action_guidance") or "None",
                    increment_action_guidance=self._format_experiences(
                        increment_task.get("increment_action_guidance", [])
                    ),
                ),
                max_tokens=384,
                temperature=0.1,
            )
            self.task_memory[task_type][status]["action_guidance"] = summary or self._format_experiences(
                increment_task.get("increment_action_guidance", [])
            )

    def _short2long(self, expert_trajectory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        increment_env: dict[str, Any] = {}
        increment_task: dict[str, Any] = {}

        env_description = expert_trajectory["env"]
        task_type = expert_trajectory["task_type"]
        status = "success" if expert_trajectory["is_success"] else "fail"
        trajectory_id = expert_trajectory["id"]

        if env_description:
            env_new = env_description not in self.env_memory
            env_slot = self.env_memory.setdefault(
                env_description,
                {"known_obs": "", "increment_traj": [], "all_traj": []},
            )
            env_slot["increment_traj"].append(trajectory_id)
            env_slot["all_traj"].append(trajectory_id)
            if env_new or len(env_slot["increment_traj"]) > self.env_batch_size:
                samples = self._get_samples(env_slot["increment_traj"])
                increment_known_obs = [
                    sample["function"]
                    for sample in samples
                    if sample.get("function") and sample.get("function") != "None"
                ]
                if increment_known_obs:
                    increment_env = {
                        "known_obs": env_slot.get("known_obs", ""),
                        "increment_known_obs": increment_known_obs,
                    }
                env_slot["increment_traj"] = []

        task_slot = self.task_memory.setdefault(task_type, {})
        task_new = status not in task_slot
        status_slot = task_slot.setdefault(
            status,
            {"action_guidance": "", "increment_traj": [], "all_traj": []},
        )
        status_slot["increment_traj"].append(trajectory_id)
        status_slot["all_traj"].append(trajectory_id)
        if task_new or len(status_slot["increment_traj"]) > self.task_batch_size:
            samples = self._get_samples(status_slot["increment_traj"])
            increment_action_guidance = [
                {
                    "task": sample.get("task", ""),
                    "my_actions": sample.get("action", ""),
                    "is_success": sample.get("is_success", False),
                    "reflection": sample.get("reflection", ""),
                }
                for sample in samples
            ]
            increment_task = {
                "task_type": task_type,
                "action_guidance": status_slot.get("action_guidance", ""),
                "increment_action_guidance": increment_action_guidance,
            }
            status_slot["increment_traj"] = []

        return increment_env, increment_task

    def _search_messages(
        self,
        query_task: str,
        label: bool,
        topk: int,
        threshold: float,
    ) -> list[MASMessage]:
        if topk <= 0 or self.memory_size == 0:
            return []
        try:
            docs = self.main_memory.similarity_search_with_score(
                query=query_task,
                k=max(topk, 1),
                filter={"label": label},
            )
        except Exception:
            return []

        messages: list[MASMessage] = []
        for doc, score in sorted(docs, key=lambda item: item[1]):
            similarity = 1 - score
            if threshold and similarity < threshold:
                continue
            try:
                messages.append(MASMessage.from_dict(doc.metadata))
            except Exception:
                continue
            if len(messages) >= topk:
                break
        return messages

    def _recall_task_guidance(self, task_type: str, max_len: int = 6) -> str:
        if task_type not in self.task_memory:
            return ""

        lines: list[str] = []
        for status in ("success", "fail"):
            if status not in self.task_memory[task_type]:
                continue
            summary = self.task_memory[task_type][status].get("action_guidance", "")
            for item in self._split_summary(summary):
                prefix = "Success" if status == "success" else "Failure"
                lines.append(f"{prefix}: {item}")
                if len(lines) >= max_len:
                    return "\n".join(lines)
        return "\n".join(lines)

    @staticmethod
    def _recall_to_guidance(recall: dict[str, Any]) -> list[str]:
        guidance: list[str] = []
        if recall.get("known_obs"):
            guidance.append("Known environment observations:\n" + recall["known_obs"])
        if recall.get("action_guidance"):
            guidance.append("Action guidance:\n" + recall["action_guidance"])
        if recall.get("local_reflections"):
            guidance.append("Past local reflections:\n" + "\n".join(recall["local_reflections"]))
        return guidance

    def _ensure_local_slot(self, key: str) -> dict[str, Any]:
        return self.local_memory.setdefault(
            key,
            {
                "name": key,
                "reflection": [],
                "is_success": False,
                "skip": False,
            },
        )

    def _get_local_reflections(self) -> list[str]:
        if not self.current_task_key:
            return []
        slot = self.local_memory.get(self.current_task_key, {})
        return slot.get("reflection", [])[-self.local_reflection_limit :]

    def _get_samples(self, trajectory_ids: list[str]) -> list[dict[str, Any]]:
        return [
            self.trajectory_store[trajectory_id]
            for trajectory_id in trajectory_ids
            if trajectory_id in self.trajectory_store
        ]

    def _persist(self) -> None:
        write_json(self.local_memory, self.local_memory_path)
        write_json(self.env_memory, self.env_memory_path)
        write_json(self.task_memory, self.task_memory_path)
        write_json(self.trajectory_store, self.trajectory_store_path)

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        try:
            return self.llm_model(
                [Message("system", system_prompt), Message("user", user_prompt)],
                max_tokens=max_tokens,
                temperature=temperature,
            ).strip()
        except Exception as exc:
            print(f"CDMem memory LLM call failed: {exc}")
            return ""

    @staticmethod
    def _document_text(mas_message: MASMessage, expert_trajectory: dict[str, Any]) -> str:
        return "\n".join(
            [
                mas_message.task_main or "",
                expert_trajectory.get("task", ""),
                expert_trajectory.get("action", ""),
                expert_trajectory.get("reflection", ""),
            ]
        )

    @staticmethod
    def _split_alfworld_description(text: str) -> tuple[str, str]:
        text = text or ""
        env_description = ""
        task_description = ""

        env_match = re.search(
            r"You are in the middle of a room\..*?(?=\n\s*\n|Your task is to:|$)",
            text,
            re.DOTALL,
        )
        if env_match:
            env_description = env_match.group(0).strip()

        task_match = re.search(r"Your task is to:\s*(.*)", text, re.DOTALL)
        if task_match:
            task_description = task_match.group(1).strip()
            task_description = task_description.split("___")[0].strip()
            task_description = task_description.split("- Environment feedback")[0].strip()
        else:
            task_description = text.strip()

        return env_description, task_description

    @staticmethod
    def _convert_task_description(task_description: str) -> str:
        text = (task_description or "").lower()
        if "find two" in text or "put two" in text or "two " in text:
            return "pick_two_obj"
        if "look at" in text or "examine" in text:
            return "look_at_obj"
        if "put" in text:
            if "heat" in text or "hot" in text:
                return "pick_heat_then_place"
            if "clean" in text:
                return "pick_clean_then_place"
            if "cool" in text:
                return "pick_cool_then_place"
            return "pick_and_place"
        return "unknown"

    @staticmethod
    def _parse_expert_result(expert_result: str) -> tuple[str, str, str]:
        text = expert_result or ""

        location = CDMemMASMemory._match_section(
            text,
            r"(?:\(1\)\s*)?Locations?\s*:\s*(.*?)(?=\n\s*(?:\(2\)\s*)?Functions?\s*:|\n\s*Expert Actions\s*:|\Z)",
        )
        function = CDMemMASMemory._match_section(
            text,
            r"(?:\(2\)\s*)?Functions?\s*:\s*(.*?)(?=\n\s*Expert Actions\s*:|\Z)",
        )
        action = CDMemMASMemory._match_section(
            text,
            r"Expert Actions\s*:\s*(.*)\Z",
        )
        return location, function, action

    @staticmethod
    def _match_section(text: str, pattern: str) -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_reflection(reflection_result: str) -> str:
        match = re.search(r"Reflection\s*:\s*(.*)", reflection_result or "", flags=re.IGNORECASE | re.DOTALL)
        if match:
            return "Reflection: " + match.group(1).strip()
        return (reflection_result or "").strip()

    @staticmethod
    def _fallback_reflection(is_success: bool, action: str) -> str:
        if is_success:
            return "Reflection: Reuse the critical action sequence that completed the task: " + action
        return "Reflection: The previous attempt failed; avoid repeating invalid operations and revise the search or manipulation plan."

    @staticmethod
    def _strip_thoughts(trajectory: str) -> str:
        lines = []
        for raw_line in (trajectory or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("> think:") or line.startswith("think:"):
                continue
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _split_summary(summary: str) -> list[str]:
        items: list[str] = []
        for raw_line in (summary or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\.)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if line:
                items.append(line)
        return items

    @staticmethod
    def _format_list(items: list[str]) -> str:
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))

    @staticmethod
    def _format_experiences(experiences: list[dict[str, Any]]) -> str:
        chunks = []
        for i, experience in enumerate(experiences, 1):
            chunks.append(
                "\n".join(
                    [
                        f"Experience {i}:",
                        f"Task: {experience.get('task', '')}",
                        f"Actions: {experience.get('my_actions', '')}",
                        f"Success: {experience.get('is_success', False)}",
                        f"Reflection: {experience.get('reflection', '')}",
                    ]
                )
            )
        return "\n\n".join(chunks)
