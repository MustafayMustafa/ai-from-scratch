import inspect


def get_definitions(module):
    definitions_map = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        definitions_map[name] = obj

    return definitions_map


def get_callable(name: str, module):
    registry = get_definitions(module)
    if name not in registry:
        raise ValueError(f"Function '{name}' not found in module '{module.__name__}'")

    return registry[name]
