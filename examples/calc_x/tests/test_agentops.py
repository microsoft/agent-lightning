# Copyright (c) Microsoft. All rights reserved.

import agentops
from agentops.sdk.decorators import operation

from agentlightning.reward import reward


@reward
def process_data(data: str) -> float:
    # Your function logic here
    processed_result = data.upper()  # type: ignore
    # agentops.record(Events("Processed Data", result=processed_result)) # Optional: record specific events
    return 1.0


@operation
def process_data2(data: str) -> str:
    # Your function logic here
    processed_result = data.upper()  # type: ignore
    # agentops.record(Events("Processed Data", result=processed_result)) # Optional: record specific events
    return processed_result


agentops.init()  # type: ignore
process_data("hello")
process_data2("hello2")
