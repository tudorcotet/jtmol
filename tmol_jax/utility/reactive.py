"""Memoized property decorator for frozen attrs dataclasses.

Upstream tmol's scoregraph uses a ``@reactive_property`` system to cache
derived quantities. In this port we only need the memoization: derived
tensors depend on a fully-constructed (immutable) parent, so "compute once
per instance" is enough.

Because host classes are frozen (``attrs.define(frozen=True)``), attribute
assignment is forbidden. We stash the cache on a dunder name using
``object.__setattr__`` so it bypasses the frozen check.
"""

from __future__ import annotations

from typing import Any, Callable

_CACHE_ATTR = "_reactive_cache"


class reactive_property:
    """Decorator: turn a method into a lazily-evaluated, per-instance cached property.

    Example::

        @attrs.define(frozen=True, slots=False)
        class Foo:
            x: int

            @reactive_property
            def doubled(self):
                return self.x * 2
    """

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = getattr(func, "__name__", "reactive_property")
        self.name = self.__name__

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        cache = getattr(instance, _CACHE_ATTR, None)
        if cache is None:
            cache = {}
            # bypass frozen/slots protections; requires slots=False on host class
            object.__setattr__(instance, _CACHE_ATTR, cache)
        if self.name not in cache:
            cache[self.name] = self.func(instance)
        return cache[self.name]


__all__ = ["reactive_property"]
