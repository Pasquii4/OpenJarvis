# Learning Module

The learning module implements learning policies that improve routing, agent,
and tool decisions based on historical interaction outcomes. The module provides
a `LearningPolicy` ABC taxonomy with specialized sub-ABCs for intelligence
(model routing), agent behavior, and tool selection. It also includes reward
functions for scoring inference results.

## Abstract Base Classes

### RouterPolicy

::: openjarvis.intelligence._stubs.RouterPolicy
    options:
      show_source: true
      members_order: source

### QueryAnalyzer

::: openjarvis.intelligence._stubs.QueryAnalyzer
    options:
      show_source: true
      members_order: source

### RoutingContext

`RoutingContext` is defined in [`core/types.py`](core.md#openjarvis.core.types.RoutingContext).

### RewardFunction

::: openjarvis.learning._stubs.RewardFunction
    options:
      show_source: true
      members_order: source

### LearningPolicy Taxonomy

The learning system defines a hierarchy of learning policy ABCs:

- **`LearningPolicy`** -- base ABC for all learning policies
- **`IntelligenceLearningPolicy`** -- specialization for model routing decisions
- **`AgentLearningPolicy`** -- specialization for agent behavior advice
- **`ToolLearningPolicy`** -- specialization for tool selection/configuration

---

## Policy Implementations

### TraceDrivenPolicy

::: openjarvis.learning.trace_policy.TraceDrivenPolicy
    options:
      show_source: true
      members_order: source

### classify_query

::: openjarvis.learning.trace_policy.classify_query
    options:
      show_source: true

### SFTPolicy

::: openjarvis.learning.sft_policy.SFTPolicy
    options:
      show_source: true
      members_order: source

### AgentAdvisorPolicy

::: openjarvis.learning.agent_advisor.AgentAdvisorPolicy
    options:
      show_source: true
      members_order: source

### ICLUpdaterPolicy

::: openjarvis.learning.icl_updater.ICLUpdaterPolicy
    options:
      show_source: true
      members_order: source

### GRPORouterPolicy

::: openjarvis.learning.grpo_policy.GRPORouterPolicy
    options:
      show_source: true
      members_order: source

---

## Reward Functions

### HeuristicRewardFunction

::: openjarvis.learning.heuristic_reward.HeuristicRewardFunction
    options:
      show_source: true
      members_order: source
