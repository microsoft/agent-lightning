"""Auto-tune agents with Agent Lightning.

The idea is to adopt a server-client architecture where the server holds heavy/tunable resources like
LLMs / prompts / workflow DSLs. The client(s) can request these resources and use them to orchestrate
their agent workflows. The server auto-tunes these resources based on the traces of the client(s) usage patterns.

An algorithm, implemented within an isolated program (leveraging server SDKs),
either RL or heuristic-based, receives the input of a dataset
and several specific resources, and outputs a set of tuned resources.
The algorithm runs on the server and can optionally open more ports/channels to communicate with the client(s).
The dataset is iterated over by the algorithm, and for each data sample, the algorithm can provide
a data sample as well as a set of current-versioned resources to the client(s).
The client(s) can then use these resources to perform their tasks, and the server can collect
traces of the client(s) to further tune the resources.

In current (simplified) implementation, algorithm uses Agent Lightning's server SDK to launch the server.
We do not force the algorithm to use a specific input signature. They can have their own configs, clis, etc.
We however, enforces the protocol for the algorithm to communicate with the client(s).
Algorithms use server SDK to update available resources, queue data samples, and receive traces from the client(s).
The client(s) use client SDK to request resources, perform tasks, and send traces back to the server.
There is an implicit connections between the input argument / dataset of the algorithm and the client implementation,
because the former determines the resources that the algorithm will tune, and the latter uses the resources to perform tasks.
In a platform setup, the user uses a separate interface to upload the dataset and specify algorithms' input argument (or search space?)
But we omit this for now for simplicity.
"""

from typing import TypedDict, Optional, Union, Literal
from agentlightning.client import SamplingParameters


class Resource:
    """Base class for all tunable resources."""



class LLM(Resource):

    def __init__(self, endpoint: str, model: str, sampling_parameters: SamplingParameters):
        self.endpoint = endpoint
        self.model = model
        self.sampling_parameters = sampling_parameters

    def __repr__(self):
        return f"LLM(endpoint={self.endpoint}, model={self.model}, sampling_parameters={self.sampling_parameters})"


class PromptTemplate(Resource):

    def __init__(self, template: str, engine: Literal['jinja', 'f-string', 'poml']):
        self.template = template
        self.engine = engine

    def __repr__(self):
        return f"PromptTemplate(template={self.template}, engine={self.engine})"
