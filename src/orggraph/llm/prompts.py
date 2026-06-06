"""Prompt templates for the LLM enrichment pipeline.

These are the *exact* strings sent to the model. Any change to a prompt
should be reflected in a thesis methodology section change as well.
"""


def render_emails(samples: list[dict]) -> str:
    """Render a list of sampled emails into a flat numbered block."""
    parts: list[str] = []
    for i, e in enumerate(samples, 1):
        parts.append(
            f"--- email {i} ---\n"
            f"From: {e.get('from', '')}\n"
            f"To: {e.get('to', '')}\n"
            f"Date: {e.get('date', '')}\n"
            f"Subject: {e.get('subject', '')}\n"
            f"Body:\n{e.get('body', '')}\n"
        )
    return "\n".join(parts)


PERSON_PROMPT = """You are analyzing emails from an employee at Enron Corporation. \
Based on the sampled emails below, characterize this person along several dimensions.

Return JSON only in this exact schema:
{{
  "seniority": <integer 0-10, where 0=junior analyst/assistant, 10=CEO/chairman>,
  "seniority_reasoning": "<one to two sentences explaining the seniority score>",
  "role_summary": "<one to two sentences describing what this person does>",
  "function": "<one of: Trading | Legal | Operations | GovernmentRelations | Finance | HR | IT | Pipeline | Research | Other>",
  "topics": ["<broad theme 1>", "<broad theme 2>", ...],
  "expertise": ["<narrow specialty 1>", "<narrow specialty 2>", ...],
  "persona": {{
    "formality": <0-10, 0=casual, 10=formal>,
    "verbosity": <0-10, 0=terse, 10=long>,
    "directiveness": <0-10, 0=hedging, 10=commanding>,
    "agenda_setting": <0-10, 0=responsive, 10=initiates>
  }},
  "authority_style": "<one of: directive | collaborative | facilitative | technical>",
  "communication_style": "<one to two sentences for thesis narrative>",
  "key_collaborators": ["<name 1>", "<name 2>", ...]
}}

Field constraints:
- topics: 3-7 strings, broad themes (e.g., "gas trading", "FERC regulation")
- expertise: 2-5 strings, narrow specialties (e.g., "credit derivatives", "Order 888")
- key_collaborators: 3-5 names mentioned as recurring colleagues in these emails

EMPLOYEE: {name}

EMAILS:
{emails}
"""


ENTITY_PROMPT = """You are analyzing emails between Enron Corporation and an external organization. \
Based on the sampled exchanges below, characterize this organization.

Return JSON only in this exact schema:
{{
  "description": "<one to two sentences explaining what the organization does>",
  "business_relationship": "<one of: Customer | Supplier | Partner | Regulator | LawFirm | Consultant | Government | Media | Other>",
  "industry": "<short free text, e.g., 'Investor-owned electric utility'>",
  "engagement_pattern": "<one of: transactional | strategic | regulatory | operational>",
  "tone": "<one of: cooperative | transactional | adversarial | formal | informal>",
  "topics": ["<topic 1>", "<topic 2>", ...]
}}

Field constraints:
- topics: 2-5 strings describing what they discuss with Enron

ORGANIZATION: {name} (domain: {domain})

EMAILS:
{emails}
"""


PAIR_PROMPT = """You are analyzing email exchanges between two employees at Enron Corporation. \
Based on the exchanges below, characterize their relationship.

Return JSON only in this exact schema:
{{
  "deference_score": <float -1.0 to +1.0, negative if A defers to B (B is superior), positive if B defers to A>,
  "deference_reasoning": "<one to two sentences>",
  "relationship_type": "<one of: peer | mentor | mentee | collaborator | conflict | transactional | reports_to | reports_from>",
  "shared_topics": ["<topic 1>", "<topic 2>", ...]
}}

Field constraints:
- deference_score: 0.0 means peers; the magnitude reflects how clear the asymmetry is
- shared_topics: 1-3 strings

PERSON A: {person_a}
PERSON B: {person_b}

EXCHANGES:
{emails}
"""


TOPIC_CANON_PROMPT = """You are clustering raw topical labels extracted from a corporate \
email corpus into a smaller set of canonical topics suitable for a knowledge graph.

Group the raw topics below into roughly 25-40 canonical groups. Each canonical group should \
have a short title (2-5 words) that captures the theme. Map every raw topic to exactly one \
canonical group.

Return JSON only in this exact schema:
{{
  "canonicals": {{
    "<canonical name 1>": ["<raw variant 1>", "<raw variant 2>", ...],
    "<canonical name 2>": ["<raw variant 1>", ...]
  }}
}}

RAW TOPICS:
{raw_topics}
"""
