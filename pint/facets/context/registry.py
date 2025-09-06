"""
pint.facets.context.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import functools
from collections import ChainMap
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Generic

from ..._typing import F, Magnitude
from ...compat import TypeAlias
from ...errors import UndefinedUnitError
from ...util import UnitsContainer, find_connected_nodes, find_shortest_path, split_graph_components, logger
from ..plain import GenericPlainRegistry, QuantityT, UnitDefinition, UnitT
from . import objects
from .definitions import ContextDefinition

from math import copysign

# TODO: Put back annotation when possible
# registry_cache: "RegistryCache"


class ContextCacheOverlay:
    """Layer on top of the plain UnitRegistry cache, specific to a combination of
    active contexts which contain unit redefinitions.
    """

    def __init__(self, registry_cache) -> None:
        self.dimensional_equivalents = registry_cache.dimensional_equivalents
        self.root_units = {}
        self.dimensionality = registry_cache.dimensionality
        self.parse_unit = registry_cache.parse_unit
        self.conversion_factor = {}


class GenericContextRegistry(
    Generic[QuantityT, UnitT], GenericPlainRegistry[QuantityT, UnitT]
):
    """Handle of Contexts.

    Conversion between units with different dimensions according
    to previously established relations (contexts).
    (e.g. in the spectroscopy, conversion between frequency and energy is possible)

    Capabilities:

    - Register contexts.
    - Enable and disable contexts.
    - Parse @context directive.
    """

    Context: type[objects.Context] = objects.Context

    def __init__(self, **kwargs: Any) -> None:
        # Map context name (string) or abbreviation to context.
        self._contexts: dict[str, objects.Context] = {}
        # Stores active contexts.
        self._active_ctx = objects.ContextChain()
        # Map context chain to cache
        self._caches = {}
        # Map context chain to units override
        self._context_units = {}

        super().__init__(**kwargs)

        # Allow contexts to add override layers to the units
        self._units: ChainMap[str, UnitDefinition] = ChainMap(self._units)

    def _register_definition_adders(self) -> None:
        super()._register_definition_adders()
        self._register_adder(ContextDefinition, self.add_context)

    def add_context(self, context: objects.Context | ContextDefinition) -> None:
        """Add a context object to the registry.

        The context will be accessible by its name and aliases.

        Notice that this method will NOT enable the context;
        see :meth:`enable_contexts`.
        """
        if isinstance(context, ContextDefinition):
            context = objects.Context.from_definition(context, self.get_dimensionality)

        if not context.name:
            raise ValueError("Can't add unnamed context to registry")
        if context.name in self._contexts:
            logger.warning(
                "The name %s was already registered for another context.", context.name
            )
        self._contexts[context.name] = context
        for alias in context.aliases:
            if alias in self._contexts:
                logger.warning(
                    "The name %s was already registered for another context",
                    context.name,
                )
            self._contexts[alias] = context

    def remove_context(self, name_or_alias: str) -> objects.Context:
        """Remove a context from the registry and return it.

        Notice that this methods will not disable the context;
        see :meth:`disable_contexts`.
        """
        context = self._contexts[name_or_alias]

        del self._contexts[context.name]
        for alias in context.aliases:
            del self._contexts[alias]

        return context

    def _build_cache(self, loaded_files=None) -> None:
        super()._build_cache(loaded_files)
        self._caches[()] = self._cache

    def _switch_context_cache_and_units(self) -> None:
        """If any of the active contexts redefine units, create variant self._cache
        and self._units specific to the combination of active contexts.
        The next time this method is invoked with the same combination of contexts,
        reuse the same variant self._cache and self._units as in the previous time.
        """
        del self._units.maps[:-1]
        units_overlay = any(ctx.redefinitions for ctx in self._active_ctx.contexts)
        if not units_overlay:
            # Use the default _cache and _units
            self._cache = self._caches[()]
            return

        key = self._active_ctx.hashable()
        try:
            self._cache = self._caches[key]
            self._units.maps.insert(0, self._context_units[key])
        except KeyError:
            pass

        # First time using this specific combination of contexts and it contains
        # unit redefinitions
        base_cache = self._caches[()]
        self._caches[key] = self._cache = ContextCacheOverlay(base_cache)

        self._context_units[key] = units_overlay = {}
        self._units.maps.insert(0, units_overlay)

        on_redefinition_backup = self._on_redefinition
        self._on_redefinition = "ignore"
        try:
            for ctx in reversed(self._active_ctx.contexts):
                for definition in ctx.redefinitions:
                    self._redefine(definition)
        finally:
            self._on_redefinition = on_redefinition_backup

    def _redefine(self, definition: UnitDefinition) -> None:
        """Redefine a unit from a context"""
        # Find original definition in the UnitRegistry
        candidates = self.parse_unit_name(definition.name)
        if not candidates:
            raise UndefinedUnitError(definition.name)
        candidates_no_prefix = [c for c in candidates if not c[0]]
        if not candidates_no_prefix:
            raise ValueError(f"Can't redefine a unit with a prefix: {definition.name}")
        assert len(candidates_no_prefix) == 1
        _, name, _ = candidates_no_prefix[0]
        try:
            basedef = self._units[name]
        except KeyError:
            raise UndefinedUnitError(name)

        # Rebuild definition as a variant of the plain
        if basedef.is_base:
            raise ValueError("Can't redefine a plain unit to a derived one")

        dims_old = self._get_dimensionality(basedef.reference)
        dims_new = self._get_dimensionality(definition.reference)
        if dims_old != dims_new:
            raise ValueError(
                f"Can't change dimensionality of {basedef.name} "
                f"from {dims_old} to {dims_new} in a context"
            )

        # Do not modify in place the original definition, as (1) the context may
        # be shared by other registries, and (2) it would alter the cache key
        definition = UnitDefinition(
            name=basedef.name,
            defined_symbol=basedef.symbol,
            aliases=basedef.aliases,
            reference=definition.reference,
            converter=definition.converter,
        )

        # Write into the context-specific self._units.maps[0] and self._cache.root_units
        self.define(definition)

    def enable_contexts(
        self, *names_or_contexts: str | objects.Context, **kwargs: Any
    ) -> None:
        """Enable contexts provided by name or by object.

        Parameters
        ----------
        *names_or_contexts :
            one or more contexts or context names/aliases
        **kwargs :
            keyword arguments for the context(s)

        Examples
        --------
        See :meth:`context`
        """

        # If present, copy the defaults from the containing contexts
        if self._active_ctx.defaults:
            kwargs = dict(self._active_ctx.defaults, **kwargs)

        # For each name, we first find the corresponding context
        ctxs = [
            self._contexts[name] if isinstance(name, str) else name
            for name in names_or_contexts
        ]

        # Check if the contexts have been checked first, if not we make sure
        # that dimensions are expressed in terms of plain dimensions.
        for ctx in ctxs:
            if ctx.checked:
                continue
            funcs_copy = dict(ctx.funcs)
            for (src, dst), func in funcs_copy.items():
                src_ = self._get_dimensionality(src)
                dst_ = self._get_dimensionality(dst)
                if src != src_ or dst != dst_:
                    ctx.remove_transformation(src, dst)
                    ctx.add_transformation(src_, dst_, func)
            ctx.checked = True

        # and create a new one with the new defaults.
        contexts = tuple(objects.Context.from_context(ctx, **kwargs) for ctx in ctxs)

        # Finally we add them to the active context.
        self._active_ctx.insert_contexts(*contexts)
        self._switch_context_cache_and_units()

    def disable_contexts(self, n: int | None = None) -> None:
        """Disable the last n enabled contexts.

        Parameters
        ----------
        n : int
            Number of contexts to disable. Default: disable all contexts.
        """
        self._active_ctx.remove_contexts(n)
        self._switch_context_cache_and_units()

    @contextmanager
    def context(
        self: GenericContextRegistry[QuantityT, UnitT], *names: str, **kwargs: Any
    ) -> Generator[GenericContextRegistry[QuantityT, UnitT]]:
        """Used as a context manager, this function enables to activate a context
        which is removed after usage.

        Parameters
        ----------
        *names : name(s) of the context(s).
        **kwargs : keyword arguments for the contexts.

        Examples
        --------
        Context can be called by their name:

        >>> import pint.facets.context.objects
        >>> import pint
        >>> ureg = pint.UnitRegistry()
        >>> ureg.add_context(pint.facets.context.objects.Context("one"))
        >>> ureg.add_context(pint.facets.context.objects.Context("two"))
        >>> with ureg.context("one"):
        ...     pass

        If a context has an argument, you can specify its value as a keyword argument:

        >>> with ureg.context("one", n=1):
        ...     pass

        Multiple contexts can be entered in single call:

        >>> with ureg.context("one", "two", n=1):
        ...     pass

        Or nested allowing you to give different values to the same keyword argument:

        >>> with ureg.context("one", n=1):
        ...     with ureg.context("two", n=2):
        ...         pass

        A nested context inherits the defaults from the containing context:

        >>> with ureg.context("one", n=1):
        ...     # Here n takes the value of the outer context
        ...     with ureg.context("two"):
        ...         pass
        """
        # Enable the contexts.
        self.enable_contexts(*names, **kwargs)

        try:
            # After adding the context and rebuilding the graph, the registry
            # is ready to use.
            yield self
        finally:
            # Upon leaving the with statement,
            # the added contexts are removed from the active one.
            self.disable_contexts(len(names))

    def with_context(self, name: str, **kwargs: Any) -> Callable[[F], F]:
        """Decorator to wrap a function call in a Pint context.

        Use it to ensure that a certain context is active when
        calling a function.

        Parameters
        ----------
        name :
            name of the context.
        **kwargs :
            keyword arguments for the context


        Returns
        -------
        callable: the wrapped function.

        Examples
        --------
        >>> @ureg.with_context("sp")
        ... def my_cool_fun(wavelength):
        ...     print(
        ...         "This wavelength is equivalent to: %s", wavelength.to("terahertz")
        ...     )
        """

        def decorator(func):
            assigned = tuple(
                attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr)
            )
            updated = tuple(
                attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr)
            )

            @functools.wraps(func, assigned=assigned, updated=updated)
            def wrapper(*values, **wrapper_kwargs):
                with self.context(name, **kwargs):
                    return func(*values, **wrapper_kwargs)

            return wrapper

        return decorator

    def _convert(
        self,
        value: Magnitude,
        src: UnitsContainer,
        dst: UnitsContainer,
        inplace: bool = False,
        **ctx_kwargs,
    ) -> Magnitude:
        """Convert value from some source to destination units.

        In addition to what is done by the PlainRegistry,
        converts between units with different dimensions by following
        transformation rules defined in the context.

        Parameters
        ----------
        value :
            value
        src : UnitsContainer
            source units.
        dst : UnitsContainer
            destination units.
        inplace :
             (Default value = False)
        **ctx_kwargs :
            keyword arguments for the context

        Returns
        -------
        callable
            converted value
        """
        # If there is an active context, we look for a path connecting source and
        # destination dimensionality. If it exists, we transform the source value
        # by applying sequentially each transformation of the path.
        if self._active_ctx:
            src_dim = self._get_dimensionality(src)
            dst_dim = self._get_dimensionality(dst)

            # find_shortest_path will fail if units with a transformation within
            # a context are combined with units without a transformation. For
            # example in spectroscopic context:
            # Quantity(5, 'Hz/mm').to('nm/mm', 'sp')
            # would usually fail, because the mm is not contained in the
            # transformation paths. The following block removes common dimensions
            # to src_dim and dst_dim - in the example case, the common dimension
            # of 1 [length] is removed.
            intersect = src_dim.keys() & dst_dim.keys()
            to_remove = []
            while intersect:
                common_dim = intersect.pop()
                if src_dim[common_dim] == dst_dim[common_dim]:
                    to_remove.append(common_dim)
            src_dim, dst_dim = src_dim.remove(to_remove), dst_dim.remove(to_remove)

            # code for dealing with compound units
            ctx_subgraphs = split_graph_components(self._active_ctx.graph)
            _src_dim, _dst_dim = dict(src_dim), dict(dst_dim)
            pairs = []


            # for each "base" dimension in source dimensions (i.e. without exponents)
            while _src_dim:
                base_sd = next(iter(_src_dim)) 
                # find subgraph containing base_sd
                for subgraph in ctx_subgraphs.values(): # the graph should never be empty, so this should never fail

                    # make sets of nodes with incoming and outgoing edges, and of the base dimensions in the subgraph
                    subgraph_in, subgraph_out, subgraph_base_dims = set(), set(), set()
                    for node, targets in subgraph.items():
                        subgraph_in.add(node)
                        subgraph_base_dims.add(next(iter(node)))
                        for target in targets:
                            subgraph_out.add(target)
                            subgraph_base_dims.add(next(iter(target)))
                    
                    

                    if base_sd in subgraph_base_dims:
                        # find the highest exponent of base_sd that is in the subgraph
                        exp_src = _src_dim[base_sd]
                        sign_src = int(copysign(1, _src_dim[base_sd]))
                        while exp_src != 0:
                            if UnitsContainer({base_sd:exp_src}) in subgraph_in:
                                break
                            # if not, reduce exponent by 1 and try again
                            else:
                                exp_src -= sign_src
                        else:
                            # continue to next subgraph if the dimension isn't in this subgraph with this sign
                            continue

                        # do the same for destination dimensions
                        # loop through all candidate destination base dimensions - must come from same subgraph
                        for base_dd in subgraph_base_dims:
                            if base_dd != base_sd and base_dd in _dst_dim:
                                # find the highest exponent of base_sd that is in the subgraph
                                exp_dst = _dst_dim[base_dd]
                                sign_dst = int(copysign(1, _dst_dim[base_dd]))
                                while exp_dst != 0:
                                    if UnitsContainer({base_dd:exp_dst}) in subgraph_out:
                                        break
                                    # if not, reduce exponent by 1 and try again
                                    else:
                                        exp_dst -= sign_dst
                                else:
                                    # continue to next base_dd if the dimension isn't in this subgraph with this sign
                                    continue

                                # add found pair of units and exponents to pairs
                                pairs.append((UnitsContainer({base_sd:exp_src}), UnitsContainer({base_dd:exp_dst})))
                                # remove the handled dimensions from _src_dim and _dst_dim
                                _src_dim[base_sd] -= exp_src
                                _dst_dim[base_dd] -= exp_dst
                                # delete dimensions if the exponent is now zero
                                if _src_dim[base_sd] == 0:
                                    del _src_dim[base_sd]
                                if _dst_dim[base_dd] == 0:
                                    del _dst_dim[base_dd]
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                else:
                    # put None in pairs if the source unit is not in any subgraph
                    pairs.append((None, None))
                    break


            # paths for each pair of source and destination dimension
            paths = [find_shortest_path(self._active_ctx.graph, s, d) for s, d in pairs]
            src = self.Quantity(value, src)
            for path in paths:
                if not path:
                    break
                else:
                    for a, b in zip(path[:-1], path[1:]):
                        src = self._active_ctx.transform(a, b, self, src, **ctx_kwargs)
            
            value, src = src._magnitude, src._units

        return super()._convert(value, src, dst, inplace, **ctx_kwargs)

    def _get_compatible_units(
        self, input_units: UnitsContainer, group_or_system: str | None = None
    ):
        src_dim = self._get_dimensionality(input_units)

        ret = super()._get_compatible_units(input_units, group_or_system)

        if self._active_ctx:
            ret = ret.copy()  # Do not alter self._cache
            nodes = find_connected_nodes(self._active_ctx.graph, src_dim)
            if nodes:
                for node in nodes:
                    ret |= self._cache.dimensional_equivalents[node]

        return ret


class ContextRegistry(
    GenericContextRegistry[objects.ContextQuantity[Any], objects.ContextUnit]
):
    Quantity: TypeAlias = objects.ContextQuantity[Any]
    Unit: TypeAlias = objects.ContextUnit
