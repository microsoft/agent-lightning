import json
from typing import TypedDict, Optional, Union, Literal, Dict, Type

from agentlightning.client import SamplingParameters


class Resource:
    """
    Base class for all tunable resources.

    This class provides a common interface for loading and dumping resources,
    facilitating their serialization and deserialization. Subclasses are
    automatically registered to enable loading from a generic data dictionary.
    """

    resource_type: str = 'resource'
    _registry: Dict[str, Type['Resource']] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically registers subclasses for the factory method."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'resource_type') and cls.resource_type != 'resource':
            cls._registry[cls.resource_type] = cls
        else:
            # This check ensures that concrete subclasses define a unique resource_type
            # We check if the class has its own dump method, which is a proxy for being a concrete class
            if 'dump' in cls.__dict__:
                 raise TypeError(f"Class {cls.__name__} must have a unique 'resource_type' attribute.")

    @classmethod
    def load(cls, data: Union[str, dict]) -> 'Resource':
        """
        Load a resource from a JSON string or a dictionary.

        This factory method determines the correct resource type from the input
        data and instantiates the corresponding subclass.
        """
        if isinstance(data, str):
            try:
                resource_data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string provided: {e}")
        elif isinstance(data, dict):
            resource_data = data.copy()  # Avoid modifying the original dict
        else:
            raise TypeError(f"Data must be a dict or a JSON string, not {type(data).__name__}.")

        resource_type = resource_data.pop('resource_type', None)
        if not resource_type:
            raise ValueError("Input data must contain a 'resource_type' key.")

        subclass = cls._registry.get(resource_type)
        if not subclass:
            raise ValueError(f"Unknown resource type: '{resource_type}'. "
                             f"Known types: {list(cls._registry.keys())}")

        return subclass(**resource_data)

    def dump(self) -> dict:
        """
        Dump the resource to a dictionary.

        This method should be overridden by subclasses to serialize the
        resource's attributes into a dictionary that can be easily converted
        to JSON.
        """
        raise NotImplementedError("Subclasses must implement the dump method.")


class LLM(Resource):
    """
    Provide an LLM endpoint and model name as a resource.

    Attributes:
        endpoint (str): The URL of the LLM API endpoint.
        model (str): The identifier for the model to be used (e.g., 'gpt-4o').
        sampling_parameters (SamplingParameters): A dictionary of hyperparameters
            for model inference, such as temperature, top_p, etc.
    """
    resource_type: str = 'llm'

    def __init__(self, endpoint: str, model: str, sampling_parameters: SamplingParameters):
        self.endpoint = endpoint
        self.model = model
        self.sampling_parameters = sampling_parameters

    def __repr__(self):
        return f"LLM(endpoint='{self.endpoint}', model='{self.model}', sampling_parameters={self.sampling_parameters})"

    def dump(self) -> dict:
        """
        Dump the LLM resource to a dictionary.
        """
        return {
            'resource_type': self.resource_type,
            'endpoint': self.endpoint,
            'model': self.model,
            'sampling_parameters': self.sampling_parameters
        }


class PromptTemplate(Resource):
    """
    A prompt template as a resource.

    Attributes:
        template (str): The template string. The format depends on the engine.
        engine (Literal['jinja', 'f-string', 'poml']): The templating engine
            to use for rendering the prompt.
    """
    resource_type: str = 'prompt_template'

    def __init__(self, template: str, engine: Literal['jinja', 'f-string', 'poml']):
        if engine not in ['jinja', 'f-string', 'poml']:
            raise ValueError("Engine must be one of 'jinja', 'f-string', or 'poml'.")
        self.template = template
        self.engine = engine

    def __repr__(self):
        template_repr = self.template if len(self.template) < 50 else self.template[:47] + "..."
        return f"PromptTemplate(template='{template_repr}', engine='{self.engine}')"

    def dump(self) -> dict:
        """
        Dump the PromptTemplate resource to a dictionary.
        """
        return {
            'resource_type': self.resource_type,
            'template': self.template,
            'engine': self.engine
        }


class NamedResources(Dict[str, Resource]):
    """
    A dictionary-like class to hold named resources.

    This class acts as a container for multiple `Resource` objects, keyed by
    a unique name, and provides methods for serialization and deserialization
    of the entire collection.

    Example:
        resources = NamedResources({
            'main_llm': LLM(
                endpoint="http://localhost:8080",
                model="llama3",
                sampling_parameters={'temperature': 0.7, 'max_tokens': 100}
            ),
            'system_prompt': PromptTemplate(
                template="You are a helpful assistant.",
                engine='f-string'
            )
        })
    """
    def dump(self) -> Dict[str, dict]:
        """
        Serialize all contained resources into a dictionary.

        Each resource in the collection is dumped to its dictionary
        representation.
        """
        return {name: resource.dump() for name, resource in self.items()}

    @classmethod
    def load(cls, data: Dict[str, dict]) -> 'NamedResources':
        """
        Deserialize a dictionary into a NamedResources object.

        Args:
            data: A dictionary where keys are resource names and values
                  are dictionaries representing serialized resources.

        Returns:
            A new NamedResources instance populated with deserialized
            Resource objects.
        """
        instance = cls()
        for name, resource_data in data.items():
            instance[name] = Resource.load(resource_data)
        return instance
