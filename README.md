# agentic-architecture-research
Agentio Architecture Research - Natural Language Processing

> <ins>DESCRIPTION FROM CANVAS:</ins> This project is proposed in collaboration with the HUN-REN AI Core Team who specialize in researching the connection between Agentic AI and Science.
> Large Language Models (LLMs) are increasingly used not as standalone predictors, but as agents capable of planning, reasoning, interacting with tools, and collaborating with other agents. While model capability matters, recent research shows that agent architecture design often has a larger impact on performance than model size alone.


# Plan

## Core Research Question


> Under what conditions do multi-agent LLM architectures outperform single-agent baselines in terms of accuracy, robustness, and cost-efficiency across reasoning and interactive tasks?

## Formal definition

### **Changable variables:**

- Agent Architecture
    - Single agent
    - Planner → Executor
    - Actor → Critic
    - Reflective / Memory-based agent
- Task Domain
    - HotpotQA (reasoning)
    - MiniWoB++ (interaction)
- Difficulty Level
    - Easy
    - Medium
    - Hard

## Domains

**1. HotpotQA (Multi-hop Reasoning over Documents)**

_Task:_
Convert a multi-hop natural language question into a structured reasoning process over a set of documents, producing a final answer grounded in evidence.

- Input:
     - question
     - document set (Wikipedia passages given by HotpotQA dataset)
     - optional constraints:
         - limited context window
         - distractor documents

- Core Operations:
     - identify relevant entities and relations
     - select supporting documents
     - extract key facts (e.g., attributes, dates, relationships)
     - perform multi-step reasoning (e.g.):
        - comparison
        - aggregation
        - chaining facts
---

**2. MiniWoB++ (Interactive UI Task Execution)**

_Task:_
Convert a natural language instruction and dynamic UI state into a sequence of actions that completes a task in an interactive environment.

- Input:
    - instruction (goal-oriented task description)
    - environment state:
        - DOM / UI elements
        - visible text
        - element attributes
        - optional constraints:
            - action limits
            - time (600 seconds)

- Core Operations:
    - interpret UI structure and available elements
    - map goals to interface actions
    - execute actions while updating state
    - track progress toward completion
    - optionally detect and recover from errors

### **Measurements:**
1. Performance
    - HotpotQA
        - Exact Match (EM)
        - F1 Score
    - MiniWoB++
        - Action count
        - Reward
2. Cost
    - Total tokens used
    - Number of API calls
3. Robustness
    - Variance across runs (for miniwob only, with different seeds - its the same task, and we weight each task the same)

## Experimental requirements

To make this scientifically valid, we are going to run every experiment with the same setup:
> Each architecture must run on the same input data<br/>

> Use the same LLM tool (provided by the university via API call)<br/>

## Agents

#### A1 - Single Agent

- Single agent working on the input task
- There is no role separation

#### B1 - Planner -> Executor

 __Planner:__ Decomposes tasks into steps -  __Executor:__ executes each step sequentially

 - Planner output is structured (like a list of steps)
 - Executor must follow the plan (no improvisation)

#### B2 - Actor -> Critic

__Actor:__ Provides an answer - __Critic:__ Evaluates the correctness, and consistency

- Critic must justifiy the critique
- Revision step must use critic feedback - revision happens

#### C1 - Reflective/Memory agent

- Pretrained memories from train runs
- Loads these memories at the beggining as guardrails for the task completion

## Difficulty designs

### HotpotQA

For HotpotQA dataset the questions are tagged with a difficulty level tag: `easy`, `medium`, `hard`.</br>
[Dataset from huggingface.](https://huggingface.co/datasets/hotpotqa/hotpot_qa)</br>

### MiniWoB++

Some MiniWob++ environments come with difficulty level tag: [Enviroment List](https://miniwob.farama.org/environments/list/)</br>
However, this amount of data is not particularly enough, so we are going to select `easy`, `medium`, `hard` tasks and separate them based on our own human judgment - we added each task 3 times with different seeding.


## Final output

| Architecture | Domain | Easy | Medium |	Hard |
| --- | --- | --- | --- | --- |
| Single agent |	HotpotQA | score |	score |	score |
| Planner+Executor |	HotpotQA | score |	score |	score |
| ... |	... | 	... |	... | ... |
| Single agent |	MiniWoB++ |	score | score | score |
| ...	| ... |	...	| ... | ... |
