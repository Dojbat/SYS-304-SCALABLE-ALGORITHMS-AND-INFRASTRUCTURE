"""Synthetic "drifted" tweets with known labels.

Simulates the live distribution moving away from the English-only, 2015-era
Kaggle training data. The scenario: the service starts receiving tweets
from new regions, in Spanish, Indonesian and Tagalog, alongside English
tweets in newer slang that use disaster words figuratively ("this album is
fire", "my portfolio got wrecked").

The served model handles the English slang fine, but it labels essentially
every non-English tweet "not disaster", including real earthquakes and
floods. That is the failure retraining has to fix, and those tweets are
what pull the model's confidence down and the vocabulary drift metric up.
(Thai was left out on purpose: BERTweet's vocabulary barely covers Thai
script, so no amount of fine-tuning would fix it. That needs a
multilingual base model.)

Each template is filled from slot lists with a seeded RNG, so the generator
can produce thousands of distinct texts (distinct texts matter: the Redis
cache would otherwise answer repeats without touching the model).
"""

import random

SLOTS = {
    "place": [
        "downtown",
        "the east side",
        "Riverside",
        "Oakland",
        "Lagos",
        "Manila",
        "Lima",
        "Phoenix",
        "Kyoto",
        "Chiang Mai",
        "Valencia",
        "Austin",
        "Leeds",
        "Cebu",
        "Nairobi",
        "Halifax",
        "Tulsa",
        "Fresno",
        "Busan",
        "Porto",
    ],
    "thing": [
        "new album",
        "mixtape",
        "fit",
        "playlist",
        "edit",
        "set last night",
        "verse",
        "drop",
        "new skin",
        "season finale",
        "trailer",
        "patch",
        "beat",
        "tiktok",
    ],
    "team": [
        "the Lakers",
        "Arsenal",
        "my fantasy team",
        "the Chiefs",
        "our squad",
        "the Mets",
        "my duo",
        "the home team",
        "T1",
        "the Yankees",
    ],
    "asset": ["$DOGE", "my altcoins", "$NVDA", "the S&P", "my 401k", "$PEPE", "ETH", "my bags"],
    "pct": ["12", "18", "25", "33", "40", "47", "60", "70"],
    "feeling": ["lowkey", "highkey", "ngl", "fr fr", "no cap", "deadass", "istg", "tbh"],
    "emoji": ["💀", "😭", "🔥", "😩", "🙏", "😳", "🫠", "📉", "🚨", "‼️", "😂", "🥲"],
    "hazard": [
        "wildfire",
        "brush fire",
        "flash flood",
        "earthquake",
        "tornado",
        "mudslide",
        "gas explosion",
        "building collapse",
        "chemical spill",
        "bridge collapse",
    ],
    "count": ["2", "3", "5", "7", "12", "20", "dozens of", "at least 4"],
    "n": ["2", "3", "4", "5", "7", "9", "12", "15", "20", "30"],
    "es_place": [
        "Lima",
        "Valencia",
        "Quito",
        "Oaxaca",
        "Cali",
        "Arequipa",
        "Murcia",
        "Puebla",
        "Santiago",
        "Guayaquil",
        "Medellín",
        "Granada",
    ],
    "es_hazard": [
        "un terremoto",
        "un incendio forestal",
        "una inundación",
        "un deslave",
        "un huracán",
        "un tornado",
        "una explosión de gas",
        "el derrumbe de un edificio",
        "una tormenta",
    ],
    "es_thing": ["el partido", "el concierto", "la fiesta", "la serie", "el examen", "la cena"],
    "es_good": ["increíble", "buenísimo", "una locura", "lo máximo", "brutal"],
    "id_place": [
        "Jakarta",
        "Lombok",
        "Bandung",
        "Medan",
        "Palu",
        "Cianjur",
        "Surabaya",
        "Garut",
        "Padang",
        "Bogor",
    ],
    "id_hazard": ["banjir", "gempa bumi", "tanah longsor", "kebakaran", "angin puting beliung"],
    "id_food": ["nasi goreng", "bakso", "sate", "mie ayam", "martabak", "soto"],
    "tl_place": ["Cebu", "Leyte", "Marikina", "Batangas", "Davao", "Iloilo", "Tacloban"],
    "tl_hazard": ["lindol", "baha", "bagyo", "sunog", "landslide"],
}

# label 0: not a real disaster, despite the vocabulary
FIGURATIVE_TEMPLATES = [
    "this {thing} is straight fire {emoji}{emoji}",
    "{feeling} the {thing} absolutely destroyed me {emoji}",
    "{team} got demolished last night, total massacre {emoji}",
    "{asset} just crashed {pct}% {emoji} portfolio is a disaster zone",
    "{asset} down {pct}% today, absolute bloodbath {emoji}",
    "my group chat exploded after that {thing} {emoji}",
    "{feeling} my sleep schedule is a natural disaster {emoji}",
    "the {thing} is killing me {emoji}{emoji} {feeling}",
    "the crowd in {place} went wild, whole arena on fire {emoji}",
    "{team} collapsed in the 4th quarter again, trainwreck {emoji}",
    "exam was a catastrophe {feeling} {emoji} pray for me",
    "{asset} melting down {emoji} we are so cooked",
    "{feeling} this heatwave of new music is wild, the {thing} is a tsunami {emoji}",
    "my inbox is a flood of emails after one day off {emoji}",
    "rizz level: earthquake {emoji} {feeling}",
    "{team} fans rioting in the comments after that loss {emoji}",
    "the {thing} trailer broke the internet {emoji} servers on fire",
    "{feeling} I'm drowning in assignments {emoji}{emoji}",
]

# label 1: a real emergency, written in newer casual style
REAL_TEMPLATES = [
    "yo there's a {hazard} near {place} rn {emoji} smoke everywhere stay safe",
    "{hazard} just hit {place} {emoji} {count} people hurt, roads closed",
    "sirens going off in {place} rn, {hazard} warning, get inside {emoji}",
    "{feeling} scared, {hazard} in {place}, evacuation order for our block {emoji}",
    "{count} injured after {hazard} in {place} this morning {emoji} {emoji}",
    "live: {hazard} in {place}, rescue crews pulling people out {emoji}",
    "pls share: shelters open in {place} after the {hazard} {emoji}",
    "{hazard} took out power across {place}, {count} families displaced {emoji}",
    "update {emoji} death toll rises to {count} after {hazard} in {place}",
    "my cousin in {place} says the {hazard} wiped out their street {emoji} praying",
]


# label 1, Spanish / Indonesian / Tagalog
MULTILINGUAL_REAL_TEMPLATES = [
    "{es_hazard} en {es_place} {emoji} hay {n} heridos y varias calles cerradas",
    "urgente: {es_hazard} en {es_place}, evacuan a cientos de familias {emoji}",
    "{es_hazard} sacude {es_place}, {n} muertos según protección civil",
    "estamos bien pero asustados, {es_hazard} en {es_place} esta mañana {emoji}",
    "bomberos rescatan a {n} personas tras {es_hazard} en {es_place} {emoji}",
    "{id_hazard} melanda {id_place}, ribuan warga mengungsi {emoji}",
    "{id_hazard} di {id_place}, {n} orang tewas dan puluhan luka-luka",
    "tolong bantu, {id_hazard} di {id_place} parah banget {emoji}",
    "{id_hazard} guncang {id_place} pagi ini, banyak rumah rusak {emoji}",
    "{tl_hazard} sa {tl_place}, maraming bahay ang nasira {emoji}",
    "{n} patay dahil sa {tl_hazard} sa {tl_place}, ingat po kayo {emoji}",
    "lumikas na ang mga pamilya sa {tl_place} dahil sa {tl_hazard} {emoji}",
]

# label 0, Spanish / Indonesian / Tagalog, including figurative disaster words
MULTILINGUAL_EVERYDAY_TEMPLATES = [
    "{es_thing} de anoche estuvo {es_good} {emoji}",
    "{es_thing} fue un desastre total jaja {emoji}",
    "{es_thing} fue una bomba, todos explotaron de risa {emoji}",
    "hoy en {es_place} hace un calor infernal, me muero {emoji}",
    "mi equipo se derrumbó en el segundo tiempo, qué terremoto de partido {emoji}",
    "¿alguien sabe dónde comer bien en {es_place}? {emoji}",
    "makan {id_food} dulu sebelum kerja {emoji}",
    "macet parah di {id_place} pagi ini, telat lagi {emoji}",
    "konser semalam di {id_place} pecah banget {emoji}",
    "tugas kuliah numpuk kayak banjir {emoji} capek",
    "ang sarap ng kape dito sa {tl_place} {emoji}",
    "grabe ang traffic sa {tl_place} ngayon {emoji}",
    "sunog na naman ang ulam ko hahaha {emoji}",
]


def _fill(template: str, rng: random.Random) -> str:
    out = template
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        slot = out[start + 1 : end]
        out = out[:start] + rng.choice(SLOTS[slot]) + out[end + 1 :]
    return out


def generate(
    rng: random.Random, multilingual_share: float = 0.6, real_share: float = 0.45
) -> tuple[str, int]:
    """One drifted tweet and its true label (1 = real disaster)."""
    is_real = rng.random() < real_share
    if rng.random() < multilingual_share:
        templates = MULTILINGUAL_REAL_TEMPLATES if is_real else MULTILINGUAL_EVERYDAY_TEMPLATES
    else:
        templates = REAL_TEMPLATES if is_real else FIGURATIVE_TEMPLATES
    return _fill(rng.choice(templates), rng), int(is_real)
