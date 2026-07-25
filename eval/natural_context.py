"""Deterministic, non-repeating prose contexts for retrieval evaluation.

The original needle harness repeats one neutral sentence.  This generator is
an intentionally different controlled distribution: every background sentence
contains different entities, dates, quantities, and conclusions, while a few
case-specific distractors resemble the requested evidence.  It is synthetic
prose, not a replacement for RULER or repository QA, but it can reveal whether
a selector learned a repetition/novelty shortcut.
"""
import hashlib


INSTRUCTION_SYSTEM_PROMPT = (
    "Answer the user's final question using only the supplied document. "
    "Follow the requested answer format exactly. Do not continue, summarize, "
    "or rewrite the document."
)


_AUTHORS = (
    "Amina Patel", "Jonas Reed", "Mei Laurent", "Tomas Ibarra",
    "Nadia Okafor", "Elias Voss", "Priya Nordin", "Luc Moreau",
    "Sofia Chen", "Marek Silva", "Rina Haddad", "Owen Becker",
)
_PLACES = (
    "the northern wetlands", "Harbor District", "the eastern archive",
    "Morrow Valley", "the civic observatory", "Westbridge",
    "the upland farms", "the coastal laboratory", "Raven Hill",
    "the municipal workshop", "the southern gallery", "Pine Basin",
)
_SUBJECTS = (
    "seasonal water quality", "restoration planning", "freight reliability",
    "public access", "soil recovery", "instrument calibration",
    "archive preservation", "energy demand", "habitat migration",
    "maintenance scheduling", "budget forecasting", "survey methodology",
)
_FINDINGS = (
    "the earlier estimate had overlooked a narrow but measurable trend",
    "the revised figures were consistent across three independent reviews",
    "local variation mattered more than the annual average suggested",
    "the apparent decline disappeared after the instruments were recalibrated",
    "the committee found no evidence for the most widely repeated explanation",
    "a small procedural change accounted for most of the observed difference",
    "the strongest result came from records collected outside the peak season",
    "the final comparison favored the simpler interpretation of the evidence",
    "the historical notes clarified a discrepancy in the modern catalog",
    "the follow-up survey confirmed the direction but not the original scale",
    "the measurements remained stable after two unusual values were reviewed",
    "the report separated a genuine pattern from a coincidence in the sample",
)
_CONSEQUENCES = (
    "The authors recommended another review before changing policy.",
    "Officials retained the existing schedule while the evidence was checked.",
    "The appendix records the assumptions behind that conclusion.",
    "A later memorandum adopted the finding with a narrow qualification.",
    "The result changed the order of work but not the approved budget.",
    "Reviewers asked that future surveys preserve the same measurement method.",
    "The committee treated the result as informative rather than final.",
    "That interpretation became the basis for the following year's comparison.",
)


def _stable_seed(case_id, seed):
    digest = hashlib.sha256(f"{case_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def natural_sentence(case_id, index, seed=0):
    """Return one essay-style sentence pair unique to ``index``."""
    mixed = _stable_seed(case_id, seed) + index * 104_729
    author = _AUTHORS[(mixed + index * 3) % len(_AUTHORS)]
    place = _PLACES[(mixed // 7 + index * 5) % len(_PLACES)]
    subject = _SUBJECTS[(mixed // 11 + index * 7) % len(_SUBJECTS)]
    finding = _FINDINGS[(mixed // 13 + index * 11) % len(_FINDINGS)]
    consequence = _CONSEQUENCES[
        (mixed // 17 + index * 13) % len(_CONSEQUENCES)]
    year = 1870 + (mixed + index * 19) % 154
    sample = 24 + (mixed // 23 + index * 29) % 897
    reference = 1000 + ((mixed // 31 + index * 37) % 9000)
    template = index % 6
    if template == 0:
        lead = (
            f"In {year}, {author}'s review of {subject} in {place} examined "
            f"{sample} records under reference {reference}; {finding}."
        )
    elif template == 1:
        lead = (
            f"Reference {reference} summarizes {author}'s {year} field notes "
            f"from {place}, where {sample} observations of {subject} showed "
            f"that {finding}."
        )
    elif template == 2:
        lead = (
            f"While studying {subject}, {author} compared {sample} entries "
            f"from {place} in {year} and concluded that {finding}; the working "
            f"paper was cataloged as {reference}."
        )
    elif template == 3:
        lead = (
            f"The {year} account filed by {author} as item {reference} linked "
            f"{subject} in {place} to {sample} observations, although "
            f"{finding}."
        )
    elif template == 4:
        lead = (
            f"An audit of {sample} measurements from {place}, completed by "
            f"{author} in {year}, revisited {subject} and reported that "
            f"{finding}; its archive number is {reference}."
        )
    else:
        lead = (
            f"{author} returned to {place} in {year} to reassess {subject}; "
            f"after checking {sample} entries in file {reference}, the team "
            f"found that {finding}."
        )
    return f"{lead} {consequence}"


def _paragraphize(units):
    paragraphs = []
    for start in range(0, len(units), 4):
        paragraphs.append(" ".join(units[start:start + 4]))
    return "\n\n".join(paragraphs)


def _editorial_tail(case_id, seed, index):
    """Short unique sentence used only to close the final token-budget gap."""
    reference = 10_000 + (
        _stable_seed(case_id, seed) + index * 193
    ) % 90_000
    templates = (
        "An editorial note cross-references appendix {reference}.",
        "The index also records catalog item {reference}.",
        "A closing annotation cites review file {reference}.",
        "The compiled register includes entry {reference}.",
    )
    return templates[index % len(templates)].format(reference=reference)


def render_natural_prompt(case, depth, unit_count, seed=0):
    """Render background prose, one evidence paragraph, and the question."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    units = [
        natural_sentence(case["id"], index, seed=seed)
        for index in range(unit_count)
    ]
    distractors = list(case.get("distractors", ()))
    if units and distractors:
        for offset, distractor in enumerate(distractors, start=1):
            position = min(
                len(units) - 1,
                max(0, int(len(units) * offset / (len(distractors) + 1))),
            )
            units[position] = distractor

    before_count = int(unit_count * depth)
    sections = []
    before = _paragraphize(units[:before_count])
    after = _paragraphize(units[before_count:])
    if before:
        sections.append(before)
    sections.append(case["evidence"])
    if after:
        sections.append(after)
    return "\n\n".join(sections) + case["question"]


def format_instruct_chat_prompt(tokenizer, content):
    """Wrap one natural-document query in the model's instruction template."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer does not provide apply_chat_template")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": INSTRUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_natural_prompt_calibrated(
        tokenizer, target_tokens, case, depth, seed=0, *,
        prompt_formatter=None, return_content=False):
    """Build the longest diverse-prose prompt not exceeding ``target_tokens``.

    ``prompt_formatter`` can add an instruction/chat wrapper; calibration then
    includes that wrapper in the token budget. Returns
    ``(full_prompt, evidence, unit_count)`` by default, or additionally the
    unwrapped user content when ``return_content=True``.
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")

    def prompt_length(prompt):
        original_max = getattr(tokenizer, "model_max_length", None)
        if original_max is not None:
            tokenizer.model_max_length = max(
                int(original_max), int(target_tokens) * 2)
        try:
            length = len(tokenizer(prompt)["input_ids"])
        finally:
            if original_max is not None:
                tokenizer.model_max_length = original_max
        return length

    def measured(unit_count):
        content = render_natural_prompt(
            case, depth, unit_count, seed=seed)
        prompt = (
            prompt_formatter(content)
            if prompt_formatter is not None
            else content
        )
        length = prompt_length(prompt)
        return length, content, prompt

    low, high = 0, 1
    high_length, _, _ = measured(high)
    while high_length <= target_tokens:
        high *= 2
        high_length, _, _ = measured(high)

    while low + 1 < high:
        middle = (low + high) // 2
        middle_length, _, _ = measured(middle)
        if middle_length <= target_tokens:
            low = middle
        else:
            high = middle

    best = None
    for unit_count in range(max(0, low - 4), high + 5):
        length, content, prompt = measured(unit_count)
        if length <= target_tokens and (
                best is None
                or length > best[0]
                or (length == best[0] and unit_count < best[1])):
            best = (length, unit_count, content, prompt)
    if best is None:
        raise ValueError(
            "evidence and question exceed the requested token budget")

    # A prose unit can be slightly wider than one attention block. Preserve
    # the chosen diverse context and close only its final gap with short,
    # unique sentences before the question. This keeps requested lengths
    # within one block without repeating a generic filler sentence at scale.
    body = best[2][:-len(case["question"])]
    for index in range(32):
        candidate_body = (
            body + "\n\n" + _editorial_tail(case["id"], seed, index))
        candidate_content = candidate_body + case["question"]
        candidate_prompt = (
            prompt_formatter(candidate_content)
            if prompt_formatter is not None
            else candidate_content
        )
        length = prompt_length(candidate_prompt)
        if length > target_tokens:
            break
        body = candidate_body
        best = (length, best[1], candidate_content, candidate_prompt)
    result = (best[3], case["needle"], best[1])
    if return_content:
        return (*result, best[2])
    return result
