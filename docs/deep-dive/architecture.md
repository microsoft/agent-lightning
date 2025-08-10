# Architecture

## Overview

Agent Lightning follows a modular architecture that separates concerns into distinct components:

- **Core**: Base agent functionality and interfaces
- **Trainers**: Training algorithms and optimization
- **Runners**: Execution engines for different agent types
- **Monitors**: Telemetry and debugging tools

## Component Interaction

```
User Code
    ↓
Agent API
    ↓
Core Engine → Trainer
    ↓         ↓
Runner ← Monitor
```

## Extension Points

The architecture provides several extension points:

1. Custom agent implementations
2. Training algorithm plugins
3. Monitoring backends
4. Custom runners