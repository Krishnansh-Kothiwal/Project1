#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


import os, requests, time
from typing import Any
from mediator import *
from utils import global_param
from abc import ABC, abstractmethod

# Automatically load .env if present
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

class Base_Planner(ABC):
    """Base planner supporting local Ollama and Gemini endpoints."""

    def __init__(self):
        super().__init__()
        self.dialogue_system = ''
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = False

        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        if self.provider == "gemini":
            self.llm_model = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
            self.llm_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
            self.api_key = os.getenv("GEMINI_API_KEY")
        else:
            self.llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b")
            self.llm_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
            self.api_key = None

        self.call_count = 0
        self.max_calls = int(os.getenv("MAX_LLM_CALLS", "50"))

        print(f"[LLM] provider={self.provider} model={self.llm_model} endpoint={self.llm_url}")

    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show

    def initial_planning(self, decription, example):
        # OpenAI-compatible chat calls are stateless, so preserve the
        # task instructions locally and include them on every actual query.
        self.dialogue_system = decription + "\n" + example
        if self.provider == "gemini" and not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. In PowerShell run: "
                '$env:GEMINI_API_KEY="YOUR_KEY"'
            )

    def query_codex(self, prompt_text):
        if self.call_count >= self.max_calls:
            raise RuntimeError(
                f"LLM smoke-test call cap reached ({self.max_calls}). "
                "Increase MAX_LLM_CALLS only intentionally."
            )

        headers = {
            "Content-Type": "application/json",
        }
        if self.provider == "gemini" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": self.dialogue_system},
                {"role": "user", "content": prompt_text},
            ],
        }
        if self.provider == "gemini":
            data["reasoning_effort"] = "minimal"

        last_error = None
        for attempt in range(5):
            self.call_count += 1
            print(f"[LLM] call {self.call_count}/{self.max_calls} ({self.llm_model})")
            try:
                response = requests.post(
                    self.llm_url,
                    headers=headers,
                    json=data,
                    timeout=30,
                )
                if response.status_code == 429:
                    print("[LLM Rate Limit] Hit 429 quota. Backing off for 15s...")
                    time.sleep(15)
                    last_error = RuntimeError(f"HTTP 429: {response.text[:200]}")
                    continue
                elif response.status_code != 200:
                    last_error = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:1000]}"
                    )
                    print(last_error)
                    time.sleep(3)
                    continue

                result = response.json()
                time.sleep(1.0) # gentle spacing to stay under RPM limit
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                print(f"LLM request failed: {e}")
                time.sleep(3)

        raise RuntimeError(f"LLM request failed after 5 attempts: {last_error}")

    def normalize_plan_syntax(self, plan):
        if not isinstance(plan, str):
            return plan
        stripped = plan.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            normalized = "{" + stripped[1:-1] + "}"
            print(f"[LLM Normalization] Converted outer [] to {{}}: {plan!r} -> {normalized!r}")
            return normalized
        return plan

    def check_plan_isValid(self, plan):
        return isinstance(plan, str) and "{" in plan and "}" in plan

    def step_planning(self, text):
        plan = self.query_codex(text)
        plan = self.normalize_plan_syntax(plan)
        retries = 0
        while not self.check_plan_isValid(plan):
            retries += 1
            if retries >= 3:
                raise RuntimeError(f"LLM repeatedly returned an invalid plan: {plan!r}")
            print(f"{plan} is illegal Plan! Replan ...")
            plan = self.query_codex(text)
            plan = self.normalize_plan_syntax(plan)
        return plan

    @abstractmethod
    def forward(self):
        pass

class SimpleDoorKey_Planner(Base_Planner):
    def __init__(self, seed=0):
        super().__init__()
        
        self.mediator = SimpleDoorKey_Mediator()

    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)

       
    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)
        
        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill
   
    
class KeyInBox_Planner(Base_Planner):
    def __init__(self,seed=0):
        super().__init__()
        self.mediator = KeyInBox_Mediator()
    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)


    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)

        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill


class RandomBoxKey_Planner(Base_Planner):
    def __init__(self, seed=0):
        super().__init__()
        self.mediator = RandomBoxKey_Mediator()
    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
     
    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)
        
        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill
   
class ColoredDoorKey_Planner(Base_Planner):
    def __init__(self,seed=0):
        super().__init__()
        self.mediator = ColoredDoorKey_Mediator()
    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
    

    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)

        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill
   

def Planner(task,seed=0):
    if task.lower() == "simpledoorkey":
        planner = SimpleDoorKey_Planner(seed)
    elif task.lower() == "keyinbox":
        planner = KeyInBox_Planner(seed)
    elif task.lower() == "randomboxkey":
        planner = RandomBoxKey_Planner(seed)
    elif task.lower() == "coloreddoorkey":
        planner = ColoredDoorKey_Planner(seed)
    return planner
