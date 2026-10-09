"""Dependency-light contracts for consumers of model generation.

SDK output names remain static-only: importing this consumer interface must not
initialize provider SDKs. Public KG annotations resolve to GenerationProvider;
the generate method retains the provider's exact structured/streaming union.
"""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ontographrag.providers.model_providers import ModelOutput


class GenerationProvider(Protocol):
    """The three positional string arguments used by KG extraction."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        /,
    ) -> "ModelOutput": ...
