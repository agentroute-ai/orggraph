"""Scenario dataclass + a small library of hand-crafted samples.

A ``Scenario`` is the experimental unit in the RQ2 pilot. It pins
down: which Enron-derived agents participate, what brief frames the
exchange, who speaks first, and an optional opening seed message.

The full study (per thesis Ch.5 §5.3) targets 20 scenarios drawn
from real Enron events, organised by CCO Four Flows. The pilot
reported in this thesis uses 5 scenarios. Tonight's implementation
ships **one hand-crafted sample** sufficient to smoke-test the
end-to-end pipeline. Curating the remaining scenarios from the
corpus is a separate work item (see plan: "What's deferred").
"""

from __future__ import annotations

from dataclasses import dataclass

# CCO Four Flows — used to organise the scenario library so each
# experimental unit exercises a distinct organisational mechanism.
CCO_FLOWS = ("membership", "activity", "positioning", "self-structuring")


@dataclass(frozen=True)
class Scenario:
    """One experimental scenario for the RQ2 pilot.

    Attributes
    ----------
    name:
        Slug used for filenames and logging.
    flow:
        One of :data:`CCO_FLOWS`. The pilot stratifies scenario
        selection by Four Flow.
    brief:
        Natural-language scene-setting paragraph injected into both
        conditions' prompts.
    participants:
        Canonical Person names. Must all be present in the
        ``person_enrichment.csv`` loaded by the runner.
    starter:
        Canonical name of the agent that speaks first. Must appear
        in ``participants``.
    seed_message:
        Optional opening line. Empty string means the starter
        generates its own opener via the LLM.
    max_turns:
        Hard cap on dialogue length, enforced by the runner. Both
        conditions use the same value so transcripts are
        comparable in length.
    """

    name: str
    flow: str
    brief: str
    participants: tuple[str, ...]
    starter: str
    seed_message: str = ""
    max_turns: int = 8

    def __post_init__(self) -> None:
        if self.flow not in CCO_FLOWS:
            raise ValueError(
                f"Scenario {self.name}: flow {self.flow!r} not in {CCO_FLOWS}"
            )
        if self.starter not in self.participants:
            raise ValueError(
                f"Scenario {self.name}: starter {self.starter!r} "
                f"not in participants {self.participants!r}"
            )
        if len(set(self.participants)) != len(self.participants):
            raise ValueError(
                f"Scenario {self.name}: duplicate participants {self.participants!r}"
            )
        if self.max_turns < 1:
            raise ValueError(
                f"Scenario {self.name}: max_turns must be ≥ 1, got {self.max_turns}"
            )


# --- Sample scenario for tonight's smoke test ---------------------------

# Picked because both participants exist in person_enrichment.csv
# (verified in the runner's preflight) and have rich personas; the
# topic exercises the "activity coordination" CCO Flow without
# requiring a full corpus-derived event reconstruction.
SAMPLE_ACTIVITY_COORDINATION = Scenario(
    name="activity_coordination_q4_review",
    flow="activity",
    brief=(
        "It is late October 2001. Enron's general-counsel office must "
        "schedule a Q4 review of pending derivatives confirmations. "
        "Three weeks of confirmations have backed up because of regulatory "
        "uncertainty on a counterparty deal. The agenda must be set "
        "before the next risk committee meeting on Friday."
    ),
    participants=("Sara Shackleton", "Louise Kitchen"),
    starter="Sara Shackleton",
    seed_message="",  # let the starter open
    max_turns=6,
)

SAMPLE_ACTIVITY_CAPACITY_REVISIONS = Scenario(
    name="activity_capacity_revisions",
    flow="activity",
    brief=(
        "It is mid-November 2001. Two natural gas deal transfers are "
        "pending review at Enron North America. Each touches "
        "Transwestern Pipeline's California capacity book and requires "
        "the pipeline operations team to confirm available delivery "
        "capacity before the deal transfers can be executed. Operations "
        "needs to provide a capacity snapshot before the November 25 "
        "commercial review cycle."
    ),
    participants=("Sandra Brawner", "Michelle Lokay"),
    starter="Sandra Brawner",
    seed_message="",
    max_turns=6,
)

SAMPLE_MEMBERSHIP_RAPP_TRANSITION = Scenario(
    name="membership_rapp_ets_transition",
    flow="membership",
    brief=(
        "It is late October 2001. Bill Rapp has transferred from the "
        "Enron Energy Services legal department to Enron Transportation "
        "Services as gas pipeline legal counsel. Three regulatory "
        "matters from EES are still in flight, and there is a question "
        "of which matters travel with Bill to ETS and which remain at "
        "EES. Drew Fossum, as General Counsel for ETS, needs to confirm "
        "which ETS matters Bill will own and how the in-flight EES "
        "matters will be handed off."
    ),
    participants=("Drew Fossum", "Bill Rapp"),
    starter="Drew Fossum",
    seed_message="",
    max_turns=6,
)

SAMPLE_POSITIONING_FERC_INQUIRY = Scenario(
    name="positioning_ferc_california_inquiry",
    flow="positioning",
    brief=(
        "It is mid-November 2001. The Federal Energy Regulatory "
        "Commission has issued a request for information on Enron's "
        "procedural participation in California wholesale electricity "
        "market processes during 2000 and 2001. Enron must coordinate a "
        "procedural response across its government affairs and "
        "regulatory compliance functions before the December 15 "
        "submission deadline. The coordination focuses on who answers "
        "which sub-questions, what supporting documentation each side "
        "will assemble, and what review schedule fits the deadline."
    ),
    participants=("James Steffes", "Mary C. Hain"),
    starter="James Steffes",
    seed_message="",
    max_turns=6,
)

SAMPLE_SELF_STRUCTURING_STORAGE_PRICING = Scenario(
    name="self_structuring_storage_pricing_merge",
    flow="self-structuring",
    brief=(
        "It is October 2001. The ETS Commercial organization is "
        "reviewing whether the Storage and Pricing groups should remain "
        "as separate teams or be merged under a single management lead. "
        "The merger would create a cleaner reporting line into the "
        "commercial book; the regulatory complexity around Storage "
        "might be a reason to keep it independent. Danny McCarty is "
        "canvassing senior peers, and Drew Fossum's view on whether the "
        "merge changes any regulatory-compliance footprint will weigh "
        "on the decision."
    ),
    participants=("Danny McCarty", "Drew Fossum"),
    starter="Danny McCarty",
    seed_message="",
    max_turns=6,
)


SAMPLE_SCENARIOS: dict[str, Scenario] = {
    SAMPLE_ACTIVITY_COORDINATION.name: SAMPLE_ACTIVITY_COORDINATION,
    SAMPLE_ACTIVITY_CAPACITY_REVISIONS.name: SAMPLE_ACTIVITY_CAPACITY_REVISIONS,
    SAMPLE_MEMBERSHIP_RAPP_TRANSITION.name: SAMPLE_MEMBERSHIP_RAPP_TRANSITION,
    SAMPLE_POSITIONING_FERC_INQUIRY.name: SAMPLE_POSITIONING_FERC_INQUIRY,
    SAMPLE_SELF_STRUCTURING_STORAGE_PRICING.name: SAMPLE_SELF_STRUCTURING_STORAGE_PRICING,
}
