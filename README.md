# Agl-lite: A Minimal Workable Version of Agent Lightning

This project `agl-lite` which is a minimal workable version of the popular agentic rl project `Agent Lightning` (https://github.com/microsoft/agent-lightning). 

here are some key changes: 
1. remove the dependency of `litellm` and build self-owned request gateway instead; 
2. remove the dependency of `OpenTelemetry` and the whole stack built upon it, such as the tracers of agents; instead, use gateway to collect request-response data during transfer and record them into data store; 
3. following (2), the organization of data is not based on span in opentelemetry, instead, the basic trajectory format is sequence of requests (with response);
4. use `kubenetes` as the default agent runner (`minikube` for single machine), and move the retry control from data store to k8s controller, and simplify the rollout states

There are some local environment setup and configuration needed for this project refactoring in `.local/`. 