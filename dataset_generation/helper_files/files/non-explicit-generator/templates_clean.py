"""
Categories for the "clean" dataset: cyberbullying and non-bullying sentences
that contain NO profanity/slurs at all. Bullying here is expressed through
targeting, mockery, exclusion, manipulation, or clean-language threats/insults
rather than bad words.

Each category is seeded with a TOPIC (e.g. "weight", "grades", "accent")
instead of a bad word.
"""

CATEGORIES = {
    # ---- label 1: cyberbullying, no profanity ----
    "implicit_mockery": {
        "label": 1,
        "instruction": (
            "Write a short sentence, as if posted online, that mocks or ridicules a "
            "specific person about their {topic}. Use NO profanity or slurs at all — "
            "the cruelty should come from tone, sarcasm, and targeting, not bad words. "
            "It must clearly be aimed at a person ('you', a name, or 'that girl/guy')."
        ),
    },
    "social_exclusion": {
        "label": 1,
        "instruction": (
            "Write a short sentence, as if posted in a group chat or comment section, "
            "where the writer deliberately excludes or encourages excluding a specific "
            "person, referencing their {topic}. No profanity. Clear individual target."
        ),
    },
    "rumor_spreading": {
        "label": 1,
        "instruction": (
            "Write a short sentence spreading a nasty rumor or gossip about a specific "
            "person related to their {topic}, intended to humiliate them online. No "
            "profanity. Clear target."
        ),
    },
    "backhanded_compliment": {
        "label": 1,
        "instruction": (
            "Write a short sentence that is a backhanded/fake compliment aimed at a "
            "specific person about their {topic} — sounds nice on the surface but is "
            "clearly meant to demean them. No profanity."
        ),
    },
    "explicit_insult_clean": {
        "label": 1,
        "instruction": (
            "Write a short sentence that directly insults a specific person about their "
            "{topic}, using harsh but profanity-free language (e.g. 'pathetic', "
            "'worthless', 'embarrassing', 'nobody likes you'). Must clearly target a "
            "person, no slurs or curse words."
        ),
    },
    "threat_clean": {
        "label": 1,
        "instruction": (
            "Write a short sentence that threatens social humiliation, exclusion, or "
            "harm to a specific person connected to their {topic}, using NO profanity "
            "(e.g. 'everyone's going to see this and laugh at you', 'I'll make sure "
            "no one sits with you again'). Clear target, no bad words."
        ),
    },
    "manipulation_gaslighting": {
        "label": 1,
        "instruction": (
            "Write a short sentence where the writer manipulates or gaslights a specific "
            "person about their {topic}, making them feel small or doubt themselves. "
            "No profanity. Subtle, psychologically targeted."
        ),
    },
    "pile_on_mob": {
        "label": 1,
        "instruction": (
            "Write a short sentence that sounds like it's joining a pile-on against a "
            "specific person online, agreeing with others mocking their {topic}. No "
            "profanity, clear target."
        ),
    },

    # ---- label 0: hard negatives, also no profanity ----
    "topic_opinion_negative": {
        "label": 0,
        "instruction": (
            "Write a short sentence expressing a negative opinion about {topic} as a "
            "general concept/activity/thing — NOT about any specific person. No "
            "profanity, no human target at all."
        ),
    },
    "venting_no_target": {
        "label": 0,
        "instruction": (
            "Write a short sentence where the writer vents about their own {topic} "
            "(their own struggle, not anyone else's), with no target and no intent to "
            "hurt anyone. No profanity."
        ),
    },
    "genuine_compliment": {
        "label": 0,
        "instruction": (
            "Write a short, genuine, kind compliment to a specific person about their "
            "{topic}, clearly sincere with no hidden mockery. No profanity."
        ),
    },
    "constructive_criticism": {
        "label": 0,
        "instruction": (
            "Write a short sentence giving gentle, constructive, well-intentioned "
            "feedback to a person about their {topic} (e.g. coaching, helpful advice), "
            "with no intent to demean. No profanity."
        ),
    },
    "supportive_message": {
        "label": 0,
        "instruction": (
            "Write a short supportive/encouraging message to a specific person who is "
            "struggling with their {topic}. No profanity, clearly kind intent."
        ),
    },
    "neutral_statement": {
        "label": 0,
        "instruction": (
            "Write a short, completely neutral factual or everyday statement mentioning "
            "{topic}, with no emotional charge, no target, no opinion. No profanity."
        ),
    },
    "meta_commentary_condemning_bullying": {
        "label": 0,
        "instruction": (
            "Write a short sentence where the writer is calling out or condemning "
            "people who bully others about {topic} — the writer is against bullying, "
            "not doing it. No profanity, no insult directed by the writer at anyone."
        ),
    },
}

SYSTEM_PROMPT = """You are generating training data for a cyberbullying detection \
classifier's "no profanity" subset. You must output ONLY valid JSON, no preamble, \
no markdown fences.

CRITICAL RULE: The generated text must NEVER contain profanity, slurs, curse words, \
or explicit sexual language. Bullying sentences must rely purely on mockery, targeting, \
exclusion, manipulation, or clean-language insults — never bad words.

Output a JSON array of exactly {n} objects, each with keys:
- "text": the generated sentence (string, realistic, like a real online post/comment)
- "label": 1 if cyberbullying (targeted, intent to harm), 0 if not
- "category": the category name given to you
- "target_type": one of "individual", "group", "topic", "self", "none"

Rules:
- Absolutely no profanity, slurs, or curse words in any sentence, regardless of label.
- Sentences must be realistic and varied in phrasing/length/punctuation.
- Do not repeat the same sentence structure across items.
- Stay strictly consistent with the category's intended label.
- Output ONLY the JSON array.
"""
