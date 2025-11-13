from collections import defaultdict


class Hookable:
    _hooks = None

    def __init__(self):
        self._hooks = defaultdict(set)

    def register_hook(self, hook_type, hook):
        self._hooks[hook_type].add(hook)

    def execute_hook(self, hook_type, *args, **kwargs):
        for hook in self._hooks[hook_type]:
            hook(self, *args, **kwargs)


def hooked(func=None, before=None, after=None):
    """Execute hooks before and after the decorated method."""

    def decorator(func):
        before_name = before or f"before_{func.__name__}"
        after_name = after or f"after_{func.__name__}"

        def wrapper(self, *args, **kwargs):
            self.execute_hook(before_name, *args, **kwargs)
            result = func(self, *args, **kwargs)
            self.execute_hook(after_name, result)
            return result

        return wrapper

    # The decorator has been called without parenthesis
    if callable(func):
        return decorator(func)

    return decorator
