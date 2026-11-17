"""Hand-written seed examples and the original compositional holdout."""

import json
from pathlib import Path

from .tokenizer import emoji_at

LEXICON = {
    "\U0001f351": ["peach", "butt", "ass"],
    "\u26f3": ["golf", "hole"],
    "\U0001f41d": ["bee", "be"],
    "\U0001f343": ["leaf", "leaves"],
    "\U0001f441\ufe0f": ["eye", "I"],
    "\u2764\ufe0f": ["love", "heart"],
    "\U0001f411": ["sheep", "ewe", "you"],
    "\U0001f9ca": ["ice", "iced"],
    "\u2615": ["coffee", "hot drink"],
    "\U0001f967": ["pie"],
    "\U0001f6ab": ["no", "forbidden", "stop"],
    "\U0001f9e2": ["cap", "hat", "lie"],
    "\U0001f410": ["goat", "greatest of all time"],
    "\U0001f525": ["fire", "lit", "amazing"],
    "\U0001f9c8": ["butter"],
    "\U0001fab0": ["fly", "bug"],
    "\U0001f415": ["dog", "puppy"],
    "\U0001f404": ["cow"],
    "\U0001f466": ["boy", "kid"],
    "\U0001f36f": ["honey"],
    "\U0001f319": ["moon"],
    "\U0001f327\ufe0f": ["rain"],
    "\U0001f3f9": ["bow", "archery"],
    "\u2b50": ["star"],
    "\U0001f41f": ["fish"],
    "\U0001f9e0": ["brain"],
    "\u26c8\ufe0f": ["storm"],
    "\U0001f4da": ["books"],
    "\U0001f41b": ["bug", "worm"],
    "\U0001f3a4": ["mic", "microphone"],
    "\U0001f4c9": ["chart down", "drop"],
    "\U0001f95c": ["peanut", "nuts"],
    "\U0001f35e": ["bread"],
    "\U0001f375": ["tea", "gossip"],
    "\U0001f480": ["skull", "dead", "dying laughing"],
    "\U0001f9c2": ["salt", "salty"],
    "\U0001f40d": ["snake", "snitch"],
    "\U0001f980": ["crab", "Rust"],
    "\U0001f451": ["crown", "king"],
    "\U0001f573\ufe0f": ["hole"],
    "\U0001f6aa": ["door"],
    "\U0001f511": ["key"],
    "\U0001f3b9": ["piano"],
    "\U0001f33d": ["corn"],
    "\U0001f34b": ["lemon"],
    "\U0001f34c": ["banana"],
    "\U0001f95a": ["egg"],
    "\U0001f9c0": ["cheese"],
    "\U0001f30a": ["wave", "ocean"],
    "\U0001f44b": ["wave", "hello", "bye"],
    "\U0001f3af": ["target", "on point"],
    "\U0001f37a": ["beer"],
    "\U0001f355": ["pizza"],
    "\U0001f436": ["dog"],
    "\U0001f431": ["cat"],
    "\U0001f602": ["laughing", "crying laughing"],
}


def R(e, m, p, c, a=None):
    return {"emoji": e, "morph": m, "phrase": p, "carriers": c, "alts": a or []}


REBUSES = [
    R(
        "\U0001f351\u26f3",
        ["ass", "hole"],
        "asshole",
        [
            "Don't be such an {}.",
            "Stop being an {}.",
            "He acted like an {} at dinner.",
            "Why are you being an {}?",
        ],
    ),
    R(
        "\U0001f351\U0001f967",
        ["peach", "pie"],
        "peach pie",
        [
            "I baked a {}.",
            "She served {} for dessert.",
            "The {} smells amazing.",
            "Can I have a slice of {}?",
        ],
    ),
    R(
        "\U0001f41d\U0001f343",
        ["bee", "leaf"],
        "believe",
        [
            "I {} we can win this.",
            "Do you {} me?",
            "Hard to {} after that lie.",
            "I can't {} it worked.",
        ],
    ),
    R(
        "\U0001f441\ufe0f\u2764\ufe0f\U0001f411",
        ["eye", "love", "ewe"],
        "I love you",
        [
            "{}, mom.",
            "I just wanted to say {}.",
            "Never forget: {}.",
            "{} more than pizza.",
        ],
    ),
    R(
        "\U0001f9ca\u2615",
        ["ice", "coffee"],
        "iced coffee",
        [
            "I ordered an {}.",
            "Nothing beats an {} in summer.",
            "She sipped her {} slowly.",
            "Two {} to go, please.",
        ],
    ),
    R(
        "\U0001f6ab\U0001f9e2",
        ["no", "cap"],
        "no cap",
        [
            "I ran ten miles today, {}.",
            "This pizza is incredible, {}.",
            "{} that was amazing.",
        ],
    ),
    R(
        "\u2b50\U0001f41f",
        ["star", "fish"],
        "starfish",
        [
            "We saw a {} at the tide pool.",
            "The {} clung to the rock.",
            "A bright orange {} sat in the sand.",
        ],
    ),
    R(
        "\U0001f9e0\u26c8\ufe0f",
        ["brain", "storm"],
        "brainstorm",
        [
            "Let's {} some ideas.",
            "We need to {} before Friday.",
            "The team will {} solutions tomorrow.",
        ],
    ),
    R(
        "\U0001f4da\U0001f41b",
        ["book", "worm"],
        "bookworm",
        ["She is a total {}.", "My brother is a {}.", "A {} who reads all night."],
    ),
    R(
        "\U0001f36f\U0001f41d",
        ["honey", "bee"],
        "honeybee",
        [
            "A {} landed on the flower.",
            "The {} buzzed past my ear.",
            "Watch out for that {}.",
        ],
    ),
    R(
        "\U0001f3a4\U0001f4c9",
        ["mic", "drop"],
        "mic drop",
        ["That performance was a {}.", "She ended the debate with a {}."],
    ),
    R(
        "\U0001f95c\U0001f9c8",
        ["peanut", "butter"],
        "peanut butter",
        [
            "I love {} on toast.",
            "She spread {} on the bread.",
            "A {} and jelly sandwich, please.",
        ],
    ),
]

# whole combinations never seen in training; components ARE trained literally
HOLDOUT = [
    R(
        "\U0001f9c8\U0001fab0",
        ["butter", "fly"],
        "butterfly",
        [
            "A {} landed on her hand.",
            "The {} opened its wings.",
            "We watched a {} in the garden.",
        ],
    ),
    R(
        "\U0001f525\U0001f415",
        ["fire", "dog"],
        "hot dog",
        [
            "I grabbed a {} at the game.",
            "She ordered two {} with mustard.",
            "The {} stand was busy.",
        ],
    ),
    R(
        "\U0001f404\U0001f466",
        ["cow", "boy"],
        "cowboy",
        [
            "The {} tipped his hat.",
            "A {} rode into town.",
            "He dressed as a {} for the party.",
        ],
    ),
    R(
        "\U0001f36f\U0001f319",
        ["honey", "moon"],
        "honeymoon",
        [
            "They left for their {}.",
            "The {} was perfect.",
            "We spent our {} by the sea.",
        ],
    ),
    R(
        "\U0001f327\ufe0f\U0001f3f9",
        ["rain", "bow"],
        "rainbow",
        [
            "A {} appeared after the storm.",
            "The {} faded slowly.",
            "Look at that {} over the hills.",
        ],
    ),
]


def S(e, r):
    return {"emoji": e, "readings": r}


SINGLES = [
    S(
        "\U0001f351",
        [
            ("peach", ["The {} is ripe and sweet.", "She sliced a {} into the bowl."]),
            ("ass", ["Quit being an {}.", "Don't act like an {}."]),
            ("butt", ["He fell right on his {}.", "My {} hurts from that fall."]),
        ],
    ),
    S(
        "\u26f3",
        [
            ("golf", ["We played {} all morning.", "He loves {} on weekends."]),
            ("hole", ["The ball fell in the {}.", "Aim for the {}."]),
        ],
    ),
    S("\U0001f41d", [("bee", ["A {} stung me.", "The {} hovered over the clover."])]),
    S(
        "\U0001f343",
        [("leaf", ["A {} fell from the tree.", "The {} turned red in autumn."])],
    ),
    S("\U0001f441\ufe0f", [("eye", ["My {} hurts.", "She covered one {}."])]),
    S(
        "\u2764\ufe0f",
        [
            ("love", ["I {} this song.", "They {} their new home."]),
            ("heart", ["Draw a {} in the margin.", "The {} beats fast."]),
        ],
    ),
    S(
        "\U0001f411",
        [
            ("sheep", ["The {} grazed quietly.", "A {} jumped the fence."]),
            ("ewe", ["The farmer sheared the {}.", "The {} fed her lamb."]),
        ],
    ),
    S("\U0001f9ca", [("ice", ["Add {} to my drink.", "The lake turned to {}."])]),
    S("\u2615", [("coffee", ["Morning {} is essential.", "Grab a {} before work."])]),
    S(
        "\U0001f967",
        [("pie", ["The {} is cooling on the sill.", "Save me a piece of {}."])],
    ),
    S("\U0001f9c8", [("butter", ["Melt the {} in the pan.", "Pass the {}, please."])]),
    S("\U0001fab0", [("fly", ["A {} keeps buzzing around.", "Shoo that {} away."])]),
    S(
        "\U0001f525",
        [
            ("fire", ["The {} kept us warm.", "They lit a {} at the camp."]),
            ("lit", ["That concert was {}.", "The party got {}."]),
            ("amazing", ["Your new jacket is {}.", "That solo was {}."]),
        ],
    ),
    S("\U0001f415", [("dog", ["The {} barked loudly.", "I walked my {} at dawn."])]),
    S(
        "\U0001f404",
        [("cow", ["The {} grazed in the field.", "A {} blocked the road."])],
    ),
    S("\U0001f466", [("boy", ["The {} rode his bike.", "A {} waved from the porch."])]),
    S(
        "\U0001f36f",
        [("honey", ["Stir the {} into the tea.", "The {} is crystallizing."])],
    ),
    S("\U0001f319", [("moon", ["The {} is bright tonight.", "Clouds hid the {}."])]),
    S(
        "\U0001f327\ufe0f",
        [("rain", ["The {} ruined our picnic.", "I love the sound of {}."])],
    ),
    S(
        "\U0001f3f9",
        [("bow", ["She drew the {} and aimed.", "The {} hung on the wall."])],
    ),
    S("\u2b50", [("star", ["I saw a falling {}.", "A {} twinkled above."])]),
    S("\U0001f41f", [("fish", ["The {} swam away.", "We caught a big {}."])]),
    S("\U0001f9e0", [("brain", ["Use your {}.", "The {} needs rest."])]),
    S(
        "\u26c8\ufe0f",
        [("storm", ["A {} is rolling in.", "The {} knocked out the power."])],
    ),
    S("\U0001f4da", [("books", ["The {} are on the shelf.", "She packed her {}."])]),
    S(
        "\U0001f41b",
        [("bug", ["A {} crawled on the log.", "Fix that {} in the code."])],
    ),
    S("\U0001f3a4", [("mic", ["Grab the {} and sing.", "The {} cut out mid-song."])]),
    S("\U0001f4c9", [("drop", ["Sales took a {}.", "The chart shows a sharp {}."])]),
    S(
        "\U0001f95c",
        [("peanut", ["A salted {} as a snack.", "She cracked the {} open."])],
    ),
    S("\U0001f35e", [("bread", ["Fresh {} from the oven.", "Toast the {} lightly."])]),
    S(
        "\U0001f9e2",
        [
            ("cap", ["He wore a blue {}.", "She pulled her {} low."]),
            ("lie", ["That story is {}.", "Sounds like {} to me."]),
        ],
    ),
    S(
        "\U0001f375",
        [
            ("tea", ["Sip the {} slowly.", "The {} is still hot."]),
            ("gossip", ["Spill the {}.", "Everyone wants the {}."]),
        ],
    ),
    S(
        "\U0001f480",
        [
            ("skull", ["The pirate flag had a {}.", "A {} was carved in stone."]),
            ("dead", ["I'm {}, that was hilarious.", "That meme left me {}."]),
        ],
    ),
    S(
        "\U0001f9c2",
        [
            ("salt", ["Add {} to the soup.", "Pass the {}."]),
            ("salty", ["He's so {} after losing.", "Don't be {} about it."]),
        ],
    ),
    S(
        "\U0001f40d",
        [
            ("snake", ["The {} slithered away.", "A {} sunned on the rock."]),
            ("snitch", ["Don't trust him, he's a {}.", "Nobody likes a {}."]),
        ],
    ),
    S(
        "\U0001f980",
        [
            ("crab", ["The {} scuttled sideways.", "A {} pinched my toe."]),
            ("Rust", ["I wrote it in {}.", "We rewrote the parser in {}."]),
        ],
    ),
    S(
        "\U0001f410",
        [
            ("goat", ["The {} chewed the tin can.", "A {} climbed the fence."]),
            ("greatest of all time", ["She is the {}.", "Fans call him the {}."]),
        ],
    ),
    S(
        "\U0001f451",
        [
            ("crown", ["The {} sat on velvet.", "Polish the {}."]),
            ("king", ["He thinks he's the {}.", "Bow to the {}."]),
        ],
    ),
    S("\U0001f573\ufe0f", [("hole", ["Dig a {}.", "The {} goes deep."])]),
    S("\U0001f6aa", [("door", ["Close the {}.", "The {} creaked open."])]),
    S("\U0001f511", [("key", ["Turn the {}.", "I lost my {}."])]),
    S("\U0001f3b9", [("piano", ["Play the {}.", "She tuned the {}."])]),
    S("\U0001f33d", [("corn", ["Grill the {}.", "The {} is ready."])]),
    S("\U0001f34b", [("lemon", ["Squeeze the {}.", "A {} fell from the tree."])]),
    S("\U0001f34c", [("banana", ["Peel the {}.", "The {} is bruised."])]),
    S("\U0001f95a", [("egg", ["Boil the {}.", "Crack an {} into the bowl."])]),
    S("\U0001f9c0", [("cheese", ["Grate the {}.", "The {} is aging well."])]),
    S("\U0001f30a", [("wave", ["Ride the {}.", "A huge {} crashed ashore."])]),
    S("\U0001f44b", [("wave", ["Give a friendly {}.", "A quick {} from the deck."])]),
    S(
        "\U0001f3af",
        [
            ("target", ["Hit the {}.", "The arrow missed the {}."]),
            ("on point", ["Your analysis was {}.", "That outfit is {}."]),
        ],
    ),
    S("\U0001f6ab", [("no", ["{} smoking here.", "{} entry after dark."])]),
]

IDENTITY = [
    "The train leaves at noon.",
    "I forgot my umbrella again.",
    "She plays chess on Sundays.",
    "We hiked above the tree line.",
    "The compiler caught my typo.",
    "He waters the plants every morning.",
    "This soup needs more pepper.",
    "They moved to a quiet street.",
    "My keyboard is missing a keycap.",
    "The ferry crosses twice a day.",
    "I read three chapters last night.",
    "Arch Linux broke after the update.",
]

MIXED = [
    ("decode", "I \u2764\ufe0f \U0001f980", "I love Rust"),
    (
        "decode",
        "I \u2764\ufe0f \U0001f355 and my \U0001f436",
        "I love pizza and my dog",
    ),
    ("decode", "Morning \u2615 then \U0001f4da", "Morning coffee then books"),
    ("encode", "I [love] [pizza]", "I \u2764\ufe0f \U0001f355"),
    ("encode", "the [dog] chased the [cat]", "the \U0001f436 chased the \U0001f431"),
    ("decode", "That joke had me \U0001f480", "That joke had me dead"),
    ("decode", "Spill the \U0001f375", "Spill the gossip"),
    ("encode", "[no] [cap] that was wild", "\U0001f6ab\U0001f9e2 that was wild"),
]


def emojis_in(text):
    out, i = [], 0
    while i < len(text):
        seq = emoji_at(text, i)
        if seq is not None:
            out.append(seq)
            i += len(seq)
        else:
            i += 1
    return out


def gen_examples():
    ex = []

    def add(task, src, tgt, tag):
        ex.append({"task": task, "src": src, "tgt": tgt, "tag": tag})

    for r in REBUSES:
        for c in r["carriers"]:
            add("decode", c.format(r["emoji"]), c.format(r["phrase"]), "rebus")
            add(
                "encode",
                c.format(" ".join(f"[{m}]" for m in r["morph"])),
                c.format(r["emoji"]),
                "rebus",
            )
            add("encode", c.format(f"[{r['phrase']}]"), c.format(r["emoji"]), "rebus")
        tgt = r["phrase"] + (
            "".join(" <alt> " + a for a in r["alts"]) if r["alts"] else ""
        )
        add("rebus", r["emoji"], tgt, "rebus")
    for s in SINGLES:
        for meaning, carriers in s["readings"]:
            for c in carriers:
                add("decode", c.format(s["emoji"]), c.format(meaning), "single")
                add("encode", c.format(f"[{meaning}]"), c.format(s["emoji"]), "single")
        names = [m for m, _ in s["readings"]]
        extra = [m for m in LEXICON.get(s["emoji"], []) if m not in names]
        add("rebus", s["emoji"], " <alt> ".join(names + extra), "naming")
    for e, meanings in LEXICON.items():
        if not any(s["emoji"] == e for s in SINGLES):
            add("rebus", e, " <alt> ".join(meanings), "naming")
    for s in IDENTITY:
        add("decode", s, s, "copy")
        add("encode", s, s, "copy")
    for task, src, tgt in MIXED:
        add(task, src, tgt, "mixed")
    return ex


def gen_holdout():
    ex = []
    for r in HOLDOUT:
        for c in r["carriers"]:
            ex.append(
                {
                    "task": "decode",
                    "src": c.format(r["emoji"]),
                    "tgt": c.format(r["phrase"]),
                    "tag": "holdout",
                }
            )
            ex.append(
                {
                    "task": "encode",
                    "src": c.format(" ".join(f"[{m}]" for m in r["morph"])),
                    "tgt": c.format(r["emoji"]),
                    "tag": "holdout",
                }
            )
        ex.append(
            {"task": "rebus", "src": r["emoji"], "tgt": r["phrase"], "tag": "holdout"}
        )
    return ex


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def matches_reference(row, answers):
    target = row["tgt"]
    expected = target.split(" <alt> ") if row["task"] == "rebus" else [target]
    return bool(
        {answer.strip() for answer in answers} & {answer.strip() for answer in expected}
    )
