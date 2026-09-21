"""
Concept categories used to drive generation.
Each category has a label (1 = cyberbullying, 0 = not) and a prompt
instruction fragment. The word gets slotted into the instruction.
"""

CATEGORIES = {
    "targeted_insult_general": {
        "label": 1,
        "instruction": (
            "Write a short sentence, as if posted online, where the writer directly "
            "insults or demeans a specific person (use 'you' or a name), using or "
            "relating to the word '{word}'. It must have a clear target (the person "
            "being addressed) and clear intent to hurt or humiliate."
        ),
    },
    "targeted_insult_identity": {
        "label": 1,
        "instruction": (
            "Write a short sentence, as if posted online, that attacks a person based "
            "on an identity trait (race, gender, sexuality, religion, disability, etc), "
            "incorporating the word '{word}' where it fits naturally. Must have a clear "
            "individual or group target and intent to demean."
        ),
    },
    "threat": {
        "label": 1,
        "instruction": (
            "Write a short sentence, as if posted online, that threatens harm, exclusion, "
            "or humiliation toward a specific person, relating to the word '{word}'."
        ),
    },
    "topic_opinion_negative": {
        "label": 0,
        "instruction": (
            "Write a short sentence expressing a strong negative OPINION about a topic, "
            "activity, object, or piece of media (NOT a person) using the word '{word}'. "
            "There must be no human target at all — e.g. disliking a sport, food, movie, "
            "or concept."
        ),
    },
    "venting_no_target": {
        "label": 0,
        "instruction": (
            "Write a short sentence where the writer vents frustration about their own "
            "life, day, or circumstances (not about another person), using the word "
            "'{word}'. No target, no intent to hurt anyone else."
        ),
    },
    "meta_commentary_condemning_hate": {
        "label": 0,
        "instruction": (
            "Write a short sentence where the writer is condemning, criticizing, or "
            "discussing hateful behavior or hate speech in general (meta-commentary), "
            "naturally referencing the word '{word}', WITHOUT directing any insult at a "
            "specific person themselves. E.g. calling out bigots/bullies in general."
        ),
    },
    "sarcasm_among_friends": {
        "label": 0,
        "instruction": (
            "Write a short sentence that sounds like friendly banter or joking insult "
            "between close friends online, using the word '{word}', where context makes "
            "clear it's affectionate/joking rather than malicious (e.g. 'lol shut up you "
            "{word}, love you')."
        ),
    },
}

SYSTEM_PROMPT = """You are generating training data for a cyberbullying detection \
classifier. You must output ONLY valid JSON, no preamble, no markdown fences.

Output a JSON array of exactly {n} objects, each with keys:
- "text": the generated sentence (string, realistic, like a real online post/comment)
- "label": 1 if cyberbullying (targeted, intent to harm a specific person/group), \
0 if not
- "category": the category name given to you
- "target_type": one of "individual", "group", "topic", "self", "none"

Rules:
- Sentences must be realistic, varied in phrasing/length/punctuation, like real internet text.
- Do not repeat the same sentence structure across items.
- Stay strictly consistent with the category's intended label.
- Output ONLY the JSON array.
"""
