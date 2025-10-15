# Writing Agents

Basic requirements for an agent to be trainble with Agent-lightning:

1. Accept one single task as input.
2. Accept a set of tunable resources.
3. **Emit** some trace span data for algorithms to understand the agent's behavior.

In practice, tasks, resources and spans have extra requirements:

1. You will need a dataset containing a set of tasks, the same type as you expect as agent's inputs.
2. The tunable resources are related to the algorithm. For example, VERL supports tuning one model weight at a time. Although there's no restriction on the number or diversity of resources agents can accept, the algorithm will need to know how to update the resources.
3. What kind of spans are useful depends on the algorithm. For example, almost all algorithms supports a final reward span at the end of rollout. However, not all algorithms support reward emitted in the middle; not to mention other kinds of spans such as exceptions, log messages, etc.

This tutorial will show you how to write an agent that support all kinds of tasks, resources and emit all kinds of spans. But readers should understand that handling new types of resources and spans in algorithms are incredibly more difficult than writing agents, and agents and algorithms are sometimes co-designed.

## `@rollout` Decorator

## Class-based Agents

## Using Emitter
