# Basics

Agent Lightning v1.0 consists of three main components: the **API Gateway**, the **Rollout Controller**, and the **Customized Trainer**. Together, they connect existing agents to reinforcement learning through an OpenAI-compatible endpoint, without requiring changes to the agent's interaction loop.

## Overview

<p align="center">
	<img src="../images/architecture.jpg" alt="Agent Lightning v1.0 architecture" width="75%">
</p>

The API Gateway stores the core rollouts, model endpoints, and events and provides an OpenAI-compatible model proxy for agent requests. The Rollout Controller launches and manages agent executions as local processes or Kubernetes Jobs. Finally, the Customized Trainer runs model inference and optimization on the GPU side and turns the rollout data collected by the Gateway into policy updates.

This separation provides several practical advantages:

- **Zero-code-change agent integration:** an existing agent connects by redirecting its OpenAI-compatible model endpoint to the Gateway.
- **Independent resources:** model training and agent execution can run on separate machines or clusters and scale independently.
- **Open infrastructure:** agents can run on a self-hosted Kubernetes cluster instead of requiring a commercial sandbox service.

Below, we briefly introduce each component.

## API Gateway

The API Gateway is a lightweight service at the center of Agent Lightning. It stores rollouts, model endpoints, and events. It also provides an OpenAI-compatible proxy for agents to access model inference.

<p align="center">
	<img src="../images/agentlightning-schema.jpg" alt="API Gateway objects and rollout state transitions" width="50%">
</p>

### Rollout API

A **rollout** is one execution of an agent on one input. It has a globally unique ID, an input, user-defined metadata, execution configuration, and a status:

- `QUEUING`: waiting for the Controller to start the agent;
- `RUNNING`: the agent is executing;
- `SUCCEEDED`: the execution completed successfully;
- `FAILED`: the execution ended with an error or timeout.

A rollout is not the same as a training example. Algorithms such as GRPO may create several independent rollouts from the same example so that the trainer can compare their rewards.

The trainer creates rollouts through the Rollout API. The Controller reads queued rollouts and updates their status as execution progresses. Each rollout can also contain append-only events, including:

- `model_request`, recorded automatically for each model call;
- `reward`, normally reported by the agent at the end of execution;
- custom events for diagnostics and monitoring.

Every event is associated with a specific rollout ID and is later exported as training data.

### OpenAI-compatible proxy

The Gateway also acts as a reverse proxy. The trainer registers one or more model inference endpoints, and the agent sends its model requests to a rollout-specific Gateway URL. The Gateway forwards each request to the registered model endpoint and records its prompt token IDs, response token IDs, and chosen-token log probabilities as a `model_request` event.

For example, an OpenAI Chat Completions request for a training rollout is sent to:

```text
POST /proxy/rollout/{rollout_id}/attempt/{attempt_id}/mode/train/openai/v1/chat/completions
```

The corresponding validation path uses `mode/val`. OpenAI-compatible clients can use the path through `/openai/v1` as their base URL and append `/chat/completions` normally.

Because the rollout ID is part of the proxy URL, every model call is automatically associated with the correct execution. An existing agent only needs to use the provided endpoint; it does not need to implement Agent Lightning's rollout or training logic.

## Rollout Controller

The Rollout Controller turns queued rollouts into real agent executions. It continuously reconciles rollout state in the API Gateway with the processes or Jobs it manages, and reports execution progress back to the Gateway.

<p align="center">
	<img src="../images/controller-reconciliation.jpg" alt="Controller reconciliation" width="75%">
</p>

The Controller supports two modes:

- **Local mode:** starts each rollout as a short-lived local subprocess. This mode is convenient for development and debugging when the agent and trainer dependencies can share one machine.
- **Kubernetes mode:** creates one Kubernetes Job for each rollout from a user-provided template. This mode isolates agent dependencies and supports concurrent execution on a self-hosted or on-premises cluster.

The API Gateway remains the source of truth for rollout status. If a process, Kubernetes watch, or network update is interrupted, the Controller retries reconciliation until the execution state converges.

## Customized Trainer

The Customized Trainer sits on top of `verl` and connects the training backend to the API Gateway. During each training step, it:

1. registers the current model inference endpoints;
2. creates one or more rollouts for each training input;
3. waits for enough rollouts to finish;
4. retrieves model requests, rewards, and other events;
5. converts the captured calls into `verl` training samples;
6. computes advantages and updates the policy.

The trainer also handles Agent Lightning-specific data processing. It merges consecutive model calls only when their token histories are exactly continuous, computes advantages at the rollout level, and supports rollout-level loss normalization.

## Configure the components

The following chapters describe the settings for each component. Start with the trainer to define how rollouts are created and converted into training samples, then configure the server and the Controller that execute them:

1. [Trainer Configuration](4-trainer-configuration.md)
2. [API Gateway Configuration](5-api-gateway-configuration.md)
3. [Controller Configuration](6-controller-configuration.md)
