"""OrgGraph pipeline modules.

Stage layout:

    Stage 0    corpus.filter
    Stage 1    email.embed
    Stage 1.5  email.metadata
    Stage 2a   signals.extract
    Stage 2b   clusters.discover
    Stage 2c   clusters.canonicalize
    Stage 3    signals.aggregate
    Stage 4    enrich.{persons, entities, pairs}
    Stage 5    score.recompute
    Stage 6    kg.{load, sync}
    Stage 7    kg.embed
    Stage 8    eval.{qc, rq1, run_rq1}
    Stage 9    figures

Sibling subpackages (orthogonal to the main pipeline):

    validate/  external-corpus validation (BC3, Berkeley, RQ3 eval-set builder)
    bench/     LLM concurrency + email-extraction throughput benchmarks

Every module exposes both ``main()`` for CLI use and importable functions for
notebook / test / dashboard reuse. See ``Makefile`` for stage commands and
``pyproject.toml`` ``[project.scripts]`` for ``orggraph-*`` entry points.
"""
