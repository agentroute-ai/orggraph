"""Identity resolution: map email aliases to canonical employee names."""

from orggraph.data.loader import load_employees

# Common English nickname mappings (nickname -> formal names)
NICKNAMES: dict[str, list[str]] = {
    "bill": ["william"],
    "bob": ["robert"],
    "dick": ["richard"],
    "dan": ["daniel"],
    "dave": ["david"],
    "doug": ["douglas"],
    "ed": ["edward", "edgar", "edmund"],
    "geoff": ["geoffrey", "geoffery"],
    "jim": ["james"],
    "joe": ["joseph"],
    "larry": ["lawrence"],
    "jeff": ["jeffrey", "jeffery", "geoffrey"],
    "mike": ["michael"],
    "pat": ["patrick", "patricia"],
    "rick": ["richard"],
    "rob": ["robert"],
    "ron": ["ronald"],
    "sam": ["samuel"],
    "steve": ["steven", "stephen"],
    "sue": ["susan"],
    "ted": ["theodore", "edward"],
    "tom": ["thomas"],
    "tony": ["anthony"],
    "vince": ["vincent"],
    "will": ["william"],
    "chris": ["christopher", "christine", "christina"],
    "greg": ["gregory"],
    "matt": ["matthew"],
    "phil": ["philip", "phillip"],
    "liz": ["elizabeth"],
    "kay": ["katherine", "kathryn"],
    "danny": ["daniel"],
    "barry": ["barrymore", "barnard"],
    "scott": ["scott"],
    "louise": ["louise"],
    "sara": ["sarah", "sara"],
    "tana": ["tana"],
    "tracy": ["tracy"],
    "gerald": ["gerald"],
}


def build_alias_map() -> dict[str, str]:
    """Build a mapping from email address variants to canonical employee name.

    Generates aliases from:
    - Primary email (e.g., philip.allen@enron.com)
    - Folder name pattern (e.g., allen-p -> allen-p@enron.com)
    - First.last, first_last, flast, last.first and common spelling variants
    - Folder-derived localpart patterns (e.g., folder allen-p -> phillip.allen, p.allen, etc.)
    """
    df = load_employees()
    alias_map: dict[str, str] = {}

    def add_name_aliases(given: str, family: str, name: str) -> None:
        """Generate the standard / initial / doubled / nickname alias set
        for one (given, family) pair, mapping all to ``name``.

        Last-name-only aliases use ``setdefault`` so the FIRST custodian
        with a given surname keeps the bare ``family@enron.com`` slot —
        without this, processing two Whalleys would silently overwrite
        the President with a Senior Analyst.
        """
        if not (given and family):
            return
        # Standard patterns
        for sep in [".", "_", ""]:
            alias_map[f"{given}{sep}{family}@enron.com"] = name
            alias_map[f"{family}{sep}{given}@enron.com"] = name
        # Initial patterns
        alias_map[f"{given[0]}{family}@enron.com"] = name
        alias_map[f"{family}{given[0]}@enron.com"] = name
        alias_map[f"{given[0]}.{family}@enron.com"] = name
        alias_map[f"{given[0]}_{family}@enron.com"] = name
        # Double-letter variants (philip -> phillip, etc.)
        for i in range(len(given)):
            doubled = given[:i + 1] + given[i] + given[i + 1:]
            alias_map[f"{doubled}.{family}@enron.com"] = name
        # Last name only — first claimant wins (avoids burying senior people
        # under junior ones who share a surname)
        alias_map.setdefault(f"{family}@enron.com", name)
        # Nickname variants
        for nickname, formals in NICKNAMES.items():
            if given == nickname:
                for formal in formals:
                    for sep in [".", "_", ""]:
                        alias_map[f"{formal}{sep}{family}@enron.com"] = name
            elif given in formals:
                for sep in [".", "_", ""]:
                    alias_map[f"{nickname}{sep}{family}@enron.com"] = name

    for _, row in df.iterrows():
        name = row["name"]
        email = row["email"].lower().strip()
        given = row["given_name"].lower().strip()
        family = row["family_name"].lower().strip()
        folder = row["folder_name"].lower().strip()
        additional = row.get("additional_name", "").lower().strip().rstrip(".")

        # Primary email
        alias_map[email] = name

        # Folder-based alias (allen-p@enron.com)
        if folder:
            alias_map[f"{folder}@enron.com"] = name

        # Aliases from the formal given name
        add_name_aliases(given, family, name)

        # Aliases from a "go-by" name in additionalName, e.g.
        # "Lawrence Greg Whalley" with additionalName="Greg" — the corpus
        # uses greg.whalley@enron.com, which the formal-name pass misses.
        # Skip middle-initial entries (single letter) since those are not
        # used as identity-bearing tokens.
        if additional and len(additional) > 1:
            add_name_aliases(additional, family, name)

    # Manual aliases for corpus-form addresses that none of the generative
    # rules above can produce. Carol St. Clair signed her work email as
    # ``carol.st.@enron.com`` (a trailing-dot truncation of her surname);
    # the localpart-normalisation in ``resolve_sender`` cannot recover the
    # missing "clair" segment, so the address is bound to her canonical
    # name explicitly here.
    if "Carol St. Clair" in set(df["name"].astype(str)):
        alias_map["carol.st.@enron.com"] = "Carol St. Clair"

    return alias_map


def resolve_sender(email: str, alias_map: dict[str, str]) -> str | None:
    """Resolve an email address to a canonical employee name.

    First tries exact match, then tries extracting localpart patterns.
    Returns None if the address is not a known Enron employee.
    """
    email = email.lower().strip()

    # Exact match
    if email in alias_map:
        return alias_map[email]

    # Only try further matching for @enron.com addresses
    if not email.endswith("@enron.com"):
        return None

    # Extract localpart and try partial matches
    localpart = email.split("@")[0]

    # Try localpart with common separators replaced
    for sep in [".", "_", "-"]:
        normalized = localpart.replace(sep, ".")
        candidate = f"{normalized}@enron.com"
        if candidate in alias_map:
            return alias_map[candidate]

    return None
