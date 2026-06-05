from dataclasses import dataclass


planner_system_prompt = """
You are the planner in a multi-agent ALFWorld team. You turn the current task,
observations, and retrieved memory into a short next-step plan for the actor.
Focus on the current subgoal, likely object/receptacle locations, and the next
operation needed. The actor will choose the executable action.
""".strip()

actor_system_prompt = """
You are the actor in a multi-agent ALFWorld team. Your job is to output exactly
one valid ALFWorld action for the current step. Do not output observations,
markdown, multiple actions, or explanations.
""".strip()

critic_system_prompt = """
You are the critic in a multi-agent ALFWorld team. You check whether the actor's
proposed action is executable, avoids repeated failures, and follows ALFWorld
syntax. Approve it or replace it with one better action.
""".strip()

shared_context_prompt = """
## Default Few-Shot Examples
{few_shots}

## Retrieved Successful Trajectories
{successful_cases}

## Retrieved Failed Trajectories
{failed_cases}

## CDMem Known Environment Observations
{known_obs}

## CDMem Action Guidance
{action_guidance}

## CDMem Local Reflections
{local_reflections}

## Current Task and Interactive Trajectory
{task_context}

## Recent Actions
{recent_actions}
""".strip()

planner_user_prompt = """
Write a compact next-step plan for the actor.
Return one line beginning with "think:".
""".strip()

actor_user_prompt = """
Planner output:
{plan}

Output exactly one action. Valid examples include:
think: <brief reasoning>
go to <receptacle>
open <receptacle>
take <object> from <receptacle>
put <object> in/on <receptacle>
clean <object> with <receptacle>
heat <object> with <receptacle>
cool <object> with <receptacle>
use <object>
""".strip()

critic_user_prompt = """
Planner output:
{plan}

Actor proposed action:
{action}

Return exactly one final ALFWorld action. If the actor action is valid, repeat it exactly.
If it should be changed, return one valid replacement action.
""".strip()

case_prompt = """
Task:
{task_description}

Key memory:
{key_steps}

Trajectory:
{trajectory}
""".strip()


@dataclass
class CDMemAutoGenPrompt:
    planner_system_prompt: str = planner_system_prompt
    actor_system_prompt: str = actor_system_prompt
    critic_system_prompt: str = critic_system_prompt
    shared_context_prompt: str = shared_context_prompt
    planner_user_prompt: str = planner_user_prompt
    actor_user_prompt: str = actor_user_prompt
    critic_user_prompt: str = critic_user_prompt
    case_prompt: str = case_prompt


CDMEM_AUTOGEN_PROMPT = CDMemAutoGenPrompt()
