from dataclasses import dataclass
import re

from mas.agents import Agent, Env
from mas.memory.common import AgentMessage, MASMessage
from mas.mas import MetaMAS
from mas.memory import MASMemoryBase, CDMemMASMemory
from mas.reasoning import ReasoningBase, ReasoningConfig

from .cdmem_autogen_prompt import CDMEM_AUTOGEN_PROMPT


@dataclass
class CDMemAutoGen(MetaMAS):
    """ALFWorld-only multi-agent workflow using CDMem-style memory."""

    def __post_init__(self):
        self.planner_name = "planner"
        self.actor_name = "actor"
        self.critic_name = "critic"
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=["\n"])
        self.planner_config = ReasoningConfig(temperature=0.1, max_tokens=160)
        self.critic_config = ReasoningConfig(temperature=0, max_tokens=80, stop_strs=["\n"])

    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ):
        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")

        self._successful_topk = config.get("successful_topk", 1)
        self._failed_topk = config.get("failed_topk", 1)
        self._insights_topk = config.get("insights_topk", 3)
        self._threshold = config.get("threshold", 0)
        self._critic_revise = config.get("critic_revise", True)

        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"CDMem Guidance Topk: {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Critic Revise     : {self._critic_revise}")

        planner = Agent(
            name=self.planner_name,
            role="planner",
            system_instruction=CDMEM_AUTOGEN_PROMPT.planner_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )
        actor = Agent(
            name=self.actor_name,
            role="actor",
            system_instruction=CDMEM_AUTOGEN_PROMPT.actor_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )
        critic = Agent(
            name=self.critic_name,
            role="critic",
            system_instruction=CDMEM_AUTOGEN_PROMPT.critic_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )

        self.hire([planner, actor, critic])
        self.set_env(env)
        self.meta_memory = mas_memory

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)

    def schedule(self, task_config: dict) -> tuple[float, bool]:
        if task_config.get("env_name") is None or task_config.get("task_type") is None:
            raise ValueError("CDMemAutoGen currently supports AlfWorld task configs only.")
        if task_config.get("task_main") is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get("task_description") is None:
            raise ValueError("Missing required keys `task_description` in task_config")

        task_main = task_config["task_main"]
        task_description = task_config["task_description"]
        few_shots = task_config.get("few_shots", [])

        env: Env = self.env
        planner = self.get_agent(self.planner_name)
        actor = self.get_agent(self.actor_name)
        critic = self.get_agent(self.critic_name)

        env.reset()
        if hasattr(self.meta_memory, "prepare_task"):
            self.meta_memory.prepare_task(task_config)
        self.meta_memory.init_task_context(task_main, task_description)

        successful_trajectories, failed_trajectories, guidance = self.meta_memory.retrieve_memory(
            query_task=task_main,
            task_description=task_description,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )

        recall = self._get_cdmem_recall(task_description)
        action_history: list[str] = []

        initial_context = self._build_shared_context(
            few_shots=few_shots,
            successful_trajectories=successful_trajectories,
            failed_trajectories=failed_trajectories,
            guidance=guidance,
            recall=recall,
            recent_actions=action_history,
        )
        self.notify_observers(initial_context)

        for step_idx in range(env.max_trials):
            recall = self._get_cdmem_recall(task_description)
            shared_context = self._build_shared_context(
                few_shots=few_shots,
                successful_trajectories=successful_trajectories,
                failed_trajectories=failed_trajectories,
                guidance=guidance,
                recall=recall,
                recent_actions=action_history,
            )

            plan_prompt = shared_context + "\n\n" + CDMEM_AUTOGEN_PROMPT.planner_user_prompt
            plan = self._safe_response(planner, plan_prompt, self.planner_config, fallback="Plan: inspect the current observation and make progress toward the goal.")
            planner_node_id = self._record_agent_message(planner, plan_prompt, plan, [])

            actor_prompt = shared_context + "\n\n" + CDMEM_AUTOGEN_PROMPT.actor_user_prompt.format(plan=plan)
            proposed_action = self._safe_response(actor, actor_prompt, self.reasoning_config, fallback="look")
            proposed_action = self._normalize_action(proposed_action)
            actor_node_id = self._record_agent_message(actor, actor_prompt, proposed_action, [planner_node_id])

            action = proposed_action
            critic_text = proposed_action
            if self._critic_revise:
                critic_prompt = shared_context + "\n\n" + CDMEM_AUTOGEN_PROMPT.critic_user_prompt.format(
                    plan=plan,
                    action=proposed_action,
                )
                critic_text = self._safe_response(critic, critic_prompt, self.critic_config, fallback=proposed_action)
                action = self._action_from_critic(critic_text, proposed_action)
                action = self._normalize_action(action)
                self._record_agent_message(critic, critic_prompt, critic_text, [actor_node_id])

            if self._team_stuck(action, action_history):
                action = "look"

            observation, reward, done = env.step(action)
            action_history.append(action)

            step_message = (
                f"Plan {step_idx + 1}: {plan}\n"
                f"Actor {step_idx + 1}: {proposed_action}\n"
                f"Critic {step_idx + 1}: {critic_text}\n"
                f"Act {step_idx + 1}: {action}\n"
                f"Obs {step_idx + 1}: {observation}"
            )
            self.notify_observers(step_message)

            self.meta_memory.move_memory_state(action, observation, reward=reward)

            if done:
                break

        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        return final_reward, final_done

    def _record_agent_message(
        self,
        agent: Agent,
        user_prompt: str,
        message: str,
        upstream_agent_ids: list[str],
    ) -> str:
        agent_message = AgentMessage(
            agent_name=agent.name,
            system_instruction=agent.system_instruction,
            user_instruction=user_prompt,
            message=message,
        )
        return self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=upstream_agent_ids)

    def _safe_response(
        self,
        agent: Agent,
        user_prompt: str,
        config: ReasoningConfig,
        fallback: str,
    ) -> str:
        for _ in range(3):
            try:
                response = agent.response(user_prompt, config).strip()
                if response:
                    return response
            except Exception as exc:
                print(f"Error during execution of {agent.name} agent: {exc}")
        return fallback

    def _build_shared_context(
        self,
        few_shots: list[str],
        successful_trajectories: list[MASMessage],
        failed_trajectories: list[MASMessage],
        guidance: list[str],
        recall: dict,
        recent_actions: list[str],
    ) -> str:
        successful_cases = "\n\n".join(
            self._format_case(i, traj)
            for i, traj in enumerate(successful_trajectories, 1)
        ) or "None"
        failed_cases = "\n\n".join(
            self._format_case(i, traj)
            for i, traj in enumerate(failed_trajectories, 1)
        ) or "None"

        action_guidance_parts = []
        if recall.get("action_guidance"):
            action_guidance_parts.append(recall["action_guidance"])
        action_guidance_parts.extend(guidance)
        action_guidance = "\n".join(dict.fromkeys(part for part in action_guidance_parts if part)) or "None"

        local_reflections = "\n".join(recall.get("local_reflections", [])) or "None"
        recent = "\n".join(recent_actions[-6:]) or "None"

        return CDMEM_AUTOGEN_PROMPT.shared_context_prompt.format(
            few_shots="\n\n".join(few_shots) or "None",
            successful_cases=successful_cases,
            failed_cases=failed_cases,
            known_obs=recall.get("known_obs") or "None",
            action_guidance=action_guidance,
            local_reflections=local_reflections,
            task_context=self.meta_memory.summarize(),
            recent_actions=recent,
        )

    def _format_case(self, idx: int, traj: MASMessage) -> str:
        key_steps = traj.get_extra_field("key_steps") or traj.get_extra_field("cdmem_expert") or "None"
        if isinstance(key_steps, dict):
            key_steps = key_steps.get("reflection") or key_steps.get("action") or "None"
        return f"Case {idx}:\n" + CDMEM_AUTOGEN_PROMPT.case_prompt.format(
            task_description=traj.task_description,
            key_steps=key_steps,
            trajectory=traj.task_trajectory,
        )

    def _get_cdmem_recall(self, task_description: str) -> dict:
        if isinstance(self.meta_memory, CDMemMASMemory):
            return self.meta_memory.recall_cdmem(task_description)
        return {
            "known_obs": "",
            "action_guidance": "",
            "local_reflections": [],
        }

    @staticmethod
    def _normalize_action(action: str) -> str:
        action = (action or "").strip().replace("<", "").replace(">", "")
        action = action.splitlines()[0].strip()
        action = action.replace("OK.", "").replace("OK", "").strip()

        revise_match = re.match(r"^(?:APPROVE|REVISE)\s*:\s*(.*)$", action, flags=re.IGNORECASE)
        if revise_match:
            action = revise_match.group(1).strip()

        move_match = re.match(r"^move\s+(.+?)\s+to\s+(.+)$", action, flags=re.IGNORECASE)
        if move_match:
            obj, receptacle = move_match.groups()
            action = f"put {obj.strip()} in/on {receptacle.strip()}"

        put_match = re.match(r"^put\s+(.+?)\s+(?:into|in|on)\s+(.+)$", action, flags=re.IGNORECASE)
        if put_match and " in/on " not in action:
            obj, receptacle = put_match.groups()
            action = f"put {obj.strip()} in/on {receptacle.strip()}"

        return action or "look"

    @staticmethod
    def _action_from_critic(critic_text: str, proposed_action: str) -> str:
        text = (critic_text or "").strip()
        match = re.match(r"^(?:APPROVE|REVISE)\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip().splitlines()[0].strip()
            return candidate or proposed_action
        candidate = text.splitlines()[0].strip()
        return candidate or proposed_action

    @staticmethod
    def _team_stuck(current_action: str, action_history: list[str]) -> bool:
        return (
            len(action_history) >= 2
            and current_action == action_history[-1]
            and current_action == action_history[-2]
            and current_action != "look"
        )
