# Copyright (c) Microsoft. All rights reserved.

import agentops
from agentlightning.reward import reward
from agentops.sdk.decorators import operation


@reward
def process_data(data):
    # Your function logic here
    processed_result = data.upper()
    # agentops.record(Events("Processed Data", result=processed_result)) # Optional: record specific events
    return 1.0


@operation
def process_data2(data):
    # Your function logic here
    processed_result = data.upper()
    # agentops.record(Events("Processed Data", result=processed_result)) # Optional: record specific events
    return processed_result


agentops.init()
process_data("hello")
process_data2("hello2")
