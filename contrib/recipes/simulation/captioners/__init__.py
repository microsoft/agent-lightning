import warnings

from .history import HistoryPromptBuilder


def create_prompt_builder(config):
    """
    Creates an instance of a prompt builder based on the provided configuration.
    This function initializes a prompt builder by extracting relevant configuration
    parameters. It can be extended or modified to support different types of prompt
    builders beyond just the HistoryPromptBuilder.
    Args:
        config (Config): An object containing configuration settings, which must
            include the following keys:
            - max_text_history (int): Maximum number of text history entries to retain.
    Returns:
        PromptBuilder: An instance of a prompt builder configured with the specified
            history limits and any additional parameters defined in the config.
    """

    max_history = config.get("max_history", None)
    if max_history is not None:
        warnings.warn("The 'max_history' parameter is deprecated. Please use 'max_text_history' instead.")

    max_text_history = max_history
    if max_text_history is None:
        max_text_history = config.max_text_history

    return HistoryPromptBuilder(max_text_history=max_text_history)
