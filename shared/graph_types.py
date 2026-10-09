"""Static contracts for Neo4j driver-hydrated graph properties.

These are stored node/relationship property types, not arbitrary query results.
SDK names remain static-only so importing these contracts does not initialize
Neo4j or its optional dependencies. Resolving the aliases with get_type_hints
requires the SDK names in the supplied namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from neo4j.spatial import Point
    from neo4j.time import Date, DateTime, Duration, Time

# Neo4j 5.28.2 hydration v1/v2 returns these native driver types. Duration
# exposes iso_format, but only Date, Time, and DateTime expose isoformat.
Neo4jISOValue: TypeAlias = "Date | Time | DateTime"
Neo4jPropertyScalar: TypeAlias = (
    "bool | int | float | str | bytes | Date | Time | DateTime | Duration | Point"
)
Neo4jPropertyValue: TypeAlias = "Neo4jPropertyScalar | list[Neo4jPropertyScalar]"
# A projected missing property is null even though hydrated entity maps omit it.
Neo4jProperty: TypeAlias = "Neo4jPropertyValue | None"
