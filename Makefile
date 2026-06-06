.PHONY: install dashboard restore pipeline corpus email signals clusters enrich score kg eval figures \
        validate bench clean help \
        rq2-run rq2-judge rq2-aggregate rq2-pilot

help:
	@echo "OrgGraph - make targets"
	@echo ""
	@echo "  install            Install package + dev deps + pre-commit hooks"
	@echo "  restore            Load the committed Neo4j graph export into the running container"
	@echo "  dashboard          Launch the static Streamlit explorer (reads committed exports)"
	@echo ""
	@echo "  pipeline           Run Stage 0 to Stage 9 end-to-end"
	@echo "  corpus             Stage 0:    filter raw maildir to clean parquet"
	@echo "  email              Stage 1+1.5: embed emails (Ollama) + deterministic metadata"
	@echo "  signals            Stage 2a+3: LLM speech-act extraction + dyadic aggregation"
	@echo "  clusters           Stage 2b+2c: UMAP+HDBSCAN discovery + LLM canonicalization"
	@echo "  enrich             Stage 4:    Person/Entity/Pair LLM enrichment"
	@echo "  score              Stage 5:    composite scoring"
	@echo "  kg                 Stage 6+7:  Neo4j load, sync, Person embedding"
	@echo "  eval               Stage 8:    QC + RQ1 evaluation"
	@echo "  figures            Stage 9:    render result figures"
	@echo ""
	@echo "  validate           Opt-in: BC3 / Berkeley / RQ3 external validation harnesses"
	@echo "  bench              Opt-in: LLM concurrency + email-extraction benchmarks"
	@echo "  rq2-pilot          Run + judge + aggregate the RQ2 pilot end-to-end (needs LLM_BASE_URL, LLM_MODEL)"
	@echo "  rq2-run            Stage RQ2.a: run all scenarios x 2 conditions (multi-agent + single-LLM)"
	@echo "  rq2-judge          Stage RQ2.b: LLM-as-judge over every transcript on disk"
	@echo "  rq2-aggregate      Stage RQ2.c: aggregate judge verdicts into summary.csv + summary.md"
	@echo "  clean              rm -rf outputs/*"

install:
	pip install -e ".[dev]"
	pre-commit install

restore:                                   # load the committed Neo4j export into the running container
	bash scripts/restore_neo4j.sh

dashboard:                                 # static explorer (no live DB / LLM needed)
	pip install -e ".[dashboard]"
	streamlit run dashboard/Home.py

# Default reproducible pipeline: Stage 0 to Stage 9. Validation and bench are opt-in.
pipeline: corpus email signals clusters enrich score kg eval figures

corpus:                                    # Stage 0
	python -m orggraph.pipeline.corpus.filter

email:                                     # Stage 1 + 1.5
	python -m orggraph.pipeline.email.embed
	python -m orggraph.pipeline.email.metadata

signals:                                   # Stage 2a + Stage 3
	python -m orggraph.pipeline.signals.extract
	python -m orggraph.pipeline.signals.aggregate

clusters:                                  # Stage 2b + 2c
	python -m orggraph.pipeline.clusters.discover
	python -m orggraph.pipeline.clusters.canonicalize

enrich:                                    # Stage 4
	python -m orggraph.pipeline.enrich.persons
	python -m orggraph.pipeline.enrich.entities
	python -m orggraph.pipeline.enrich.pairs

score:                                     # Stage 5
	python -m orggraph.pipeline.score.recompute

kg:                                        # Stage 6 + 7
	python -m orggraph.pipeline.kg.load
	python -m orggraph.pipeline.kg.sync
	python -m orggraph.pipeline.kg.embed

eval:                                      # Stage 8
	python -m orggraph.pipeline.eval.qc
	python -m orggraph.pipeline.eval.rq1

figures:                                   # Stage 9
	python -m orggraph.pipeline.figures

validate:
	python -m orggraph.pipeline.validate.speech_acts_bc3
	python -m orggraph.pipeline.validate.clusters_berkeley
	python -m orggraph.pipeline.validate.rq3_eval_set

bench:
	python -m orggraph.pipeline.bench.llm
	python -m orggraph.pipeline.bench.email_extraction

rq2-run:                                   # 5 scenarios x 2 conditions
	PYTHONPATH=src python scripts/02_run_rq2_pilot.py --all

rq2-judge:                                 # judge every transcript on disk
	PYTHONPATH=src python scripts/03_judge_rq2_pilot.py --all

rq2-aggregate:                             # summary.csv + summary.md
	PYTHONPATH=src python scripts/04_aggregate_rq2_pilot.py

rq2-pilot: rq2-run rq2-judge rq2-aggregate

clean:
	rm -rf outputs/*
