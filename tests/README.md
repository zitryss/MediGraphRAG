# Offline backend tests

Run from the repository root with the existing development environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -rs tests
```

`tests.offline_support` blocks Python socket connections and DNS before test collection,
disables dotenv discovery, and enables Hugging Face/Transformers offline mode. This is
an in-process safeguard, not an OS network sandbox. No test intentionally calls a live
LLM, Neo4j, cloud storage, telemetry service, or downloads a model. Adapter tests verify
query/payload contracts, **not** live Cypher/provider compatibility.

To keep reports and temporary test data outside the checkout, set `TEST_ARTIFACT_DIR`
to a writable absolute directory first, then:

```bash
mkdir -p "$TEST_ARTIFACT_DIR"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_FILE="$TEST_ARTIFACT_DIR/backend.coverage" .venv/bin/python -m coverage run -m pytest -q -rs -p no:cacheprovider --basetemp="$TEST_ARTIFACT_DIR/pytest" tests
COVERAGE_FILE="$TEST_ARTIFACT_DIR/backend.coverage" .venv/bin/python -m coverage report
COVERAGE_FILE="$TEST_ARTIFACT_DIR/backend.coverage" .venv/bin/python -m coverage json -o "$TEST_ARTIFACT_DIR/backend-coverage.json"
COVERAGE_FILE="$TEST_ARTIFACT_DIR/backend.coverage" .venv/bin/python -m coverage html -d "$TEST_ARTIFACT_DIR/html"
```

The coverage configuration includes **all 81 Python modules** in the three shipped
packages, including unimported modules. `reproducibility/arxiv_v1` is a frozen historical
snapshot and is deliberately outside the active-backend denominator. To include the
separate developer-tool contract, pass
`--source=ontographrag,experiments,shared,scripts.check_arxiv_package` to `coverage run`;
report that script separately when comparing backend percentages.

## Isolation and assertions

- The experiment fixture imports the real pipeline under a temporary module name;
  only unavailable `wandb` telemetry is replaced, with fail-closed methods. Individual
  telemetry-orchestration tests explicitly replace that boundary with recording mocks.
  Environment, `sys.path`, and temporary module names are restored on teardown.
- Legacy analysis scripts use a synthetic paper manifest; no archived result files
  are necessary. Missing `make_figures` styling is explicitly replaced only in the
  two tests that need it. Figure mathematics and Matplotlib rendering remain real.
- Tests write synthetic corpus/selection/checkpoint/figure artifacts into `tmp_path`.
  Database adapters use recording responses; no database deletion is performed.
- `test_backend_*` suites cover numeric invariants, deterministic sampling, cache
  lifetimes, parameter scopes, HTTP/CLI boundaries, negative cases, CSV/PDF/ontology
  ingestion, experiment orchestration, failure denominators, and real plot outputs.
- Two **strict xfails** document pre-existing bugs rather than codify bad behavior:
  empty CSV text becomes the document `"nan"`; API-key rejection returns HTTP 500
  instead of 401. An unexpected pass fails the suite so fixes cannot go unnoticed.

## Remaining coverage limits

Coverage is not completeness. Large gaps remain in full-metrics experiment execution,
advanced entity-first/RFGE graph traversal, ontology/coreference/reverification paths,
model-backed reranking, and API KG-building/load routes. Live integration behavior is
unverified. The standalone `generate_auroc_heatmap` script imports undeclared/uninstalled
`seaborn` and performs plotting at import time; it remains explicitly uncovered, not
excluded from coverage. Legacy YouTube chunk helpers reference unavailable timestamp
functions and are not exercised as successful paths. The default suite does not add
E2E, paid-model, real benchmark, or production-data verification.
