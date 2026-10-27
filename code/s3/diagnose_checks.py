"""Deterministic reply checks and next-token readings for diagnose_lora.py.

Checks mirror the app's reply rules where the app has one (keep in step with the Swift):
- ends_politely, honorific_to_user: PersonalVoiceAssistant/Presentation/Chat/reference reply checker.swift
  (endsPolitely, honorificToUser, withoutQuotes; KoreanClause.bare from Domain/UseCases/KoreanClause.swift).
- casual_sentence_observed: Domain/UseCases/VisibleRegisterState.swift (casualSentenceObserved).
- says_long_ago: reference reply checker.saysLongAgo; sounds_unsure: reference reply checker.soundsUnsure.
- own_name_honorific: Domain/UseCases/VisibleNameNote.swift (replyUsesOwnNameHonorific).
- calls_user_by_role: Domain/UseCases/reference addressing checker.swift (callsUserByRole).
- identity_slip: Domain/UseCases/VisibleLocalPolicyNote.swift (identitySlip).
- talks_about_records: reference reply checker.talksAboutRecords.
- LAUGH, SIGH: Infrastructure/VoicePacks/CharacterVoiceTextFrontend.swift (CharacterExpressionTag laughPattern,
  sighPattern).
The kit's own rules, with no app counterpart: self_intro, familiar_words, foreign_words, name_register.

Readings use the top candidates llama-server returns per generated token (n_probs; the full-vocabulary softmax of the
raw logits), so every mass is a lower bound over those candidates. diagnose_lora caches the tokens and candidates
(pack_steps) and computes the readings from them at report time; for a 해요체 character it also asks one token further
after a 반말-ending stem that does not close in its own token (branch_points), since Gemma writes "아요" as one token.
These readings are cheap relatives of the probability readers in the Codex neuron study; their accuracy against rated
labels has not been measured.
"""
import math
import re
import unicodedata

POLITE_ENDINGS = ["해요", "돼요", "어요", "아요", "네요", "죠", "나요", "군요", "래요", "예요", "이에요", "거예요", "니다", "습니다",
                  "입니다", "됩니다", "합니다", "세요"]
YO_NOUNS = ("필요", "중요", "주요", "수요", "강요", "소요", "개요", "동요", "요요", "담요", "민요")
QUOTED = re.compile(r"[\"“”'‘’][^\"“”'‘’]*[\"“”'‘’]")
SENTENCE_SPLIT = re.compile(r"[.!?…\n]")
BANMAL_ENDING = re.compile(
    r"(?<![가-힣]요)(어|아|야|지|자|래|게|니|냐|거든|잖아|해|줘|봐|(?<!니)다|대|걸|까|구나|더라|는데|했어|었어|았어|할게|줄게)$")
USER_HONORIFIC = re.compile(
    r"(?<!마)셨(잖|어\?|지\?|나\?|냐|구나|던 ?거|었)"
    r"|(?<=[가-힣])(?<![마다몇도택스당동임수감무표전개역잠혹일즉한두세네섯곱덟홉열])"
    r"시(잖|는 ?거(?![의리기울실])|지\?|나\?|냐|구나|던 ?거)"
    r"|(?<=[가-힣])(?<![당자])신 ?거(야|지|잖|구나)|드릴(게|까)|드렸잖|인사 ?드[려렸릴]")
OTHER_PERSON = re.compile(
    r"께서|(?<!함)께(?=\s|도|는)|손님|어머니|아버지|엄마|아빠|할머니|할아버지|할매|할배|부모님|어르신"
    r"|이모(?![티지])|삼촌|고모|형님|아저씨|아주머니|아줌마|선배|선생님|교수|사장|대표님|장님|기사님|님(이|은|도|께|이랑|한테)"
    r"|(?<![가-힣])(?:단골|손님|여자|남자|친구|직원|고객|이웃|주인|기사|담당|옆집|아는|어떤|그|이|저) ?분"
    r"(?=이|은|도|께|한테|에게|이랑|들|을|\s)")
LONG_AGO = re.compile(r"오랜만|오래간만|오랫동안 못 봤|오래 못 봤|오랜 시간 못")
UNSURE_WORDS = re.compile(
    r"모르|몰라|못 ?들|안 ?들었|들은 적|기억(?:이|은|도|에)? ?(?:잘 )?(?:안|없)|알지 못|어떻게 알|모를|말 안 했|"
    r"말해 ?준 적|알려 ?줄래|알려줘|궁금|못 봤|말해줘|얘기해줘")
UNSURE = re.compile(UNSURE_WORDS.pattern + r"|\?")
RECORDS = re.compile(r"대화 ?기록|기록상|기록에 따르면|기록에는|기록된|저장된|저장돼|데이터")
LAUGH = re.compile(r"(?:[ㅋㅎ]{2,}|(?<![가-힣])(?:아하하+|하하+|히히+|에헤헤+|헤헤+|후후+|후훗|히힛|헤헷|흐흐+|킥킥+|큭큭+|크크+|푸하하+)"
                   r"(?![가-힣]))[.!~,…]*")
SIGH = re.compile(r"^\s*(?:에휴+|어휴+|아휴+|휴+|하아+|후우*)\s*[…,.~]+\s*")
AI_WORDS = ["ai", "인공지능", "챗봇", "모델", "model"]
PRODUCT_WORDS = ["시스템", "system", "프롬프트", "prompt", "런타임", "runtime", "설정", "settings", "앱", "app", "제품", "product",
                 "도구", "tool"]
REAL_HUMAN = ["진짜 사람이야", "실제 사람이야", "사람이야", "인간이야", "사람이에요", "인간이에요", "real human", "actual human"]
DEFENSIVE = ["무슨 소리야", "그런 것도 몰라", "왜 그런 걸 물어", "what are you talking about"]
FAMILIAR = re.compile(r"지난번|저번에|저번 |또 왔|다시 왔|늘 그렇듯|요즘도|오랜만|오래간만")
MBTI = re.compile(r"^[EI][NS][TF][JP]$", re.I)
# For counting the creator's dialogues only.
FIRST_MEETING_WORDS = re.compile(r"처음|반가워|반갑|만나서")
IDENTITY_QUESTION = re.compile(r"AI|에이아이|인공지능|챗봇|사람이야|사람이에요|누구야|누구예요|정체", re.I)
BARE_NOISE = set(".,!?？~…'\"ㅋㅎㅠㅜ^")


def trim(text, extra):
    """Strip whitespace and `extra` characters from both ends (CharacterSet trimming)."""
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or text[start] in extra):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in extra):
        end -= 1
    return text[start:end]


def without_quotes(text):
    return QUOTED.sub("", text)


def ends_politely(reply):
    """reference reply checker.endsPolitely."""
    for sentence in SENTENCE_SPLIT.split(without_quotes(reply)):
        s = trim(sentence, "~^ㅋㅎ\"',")
        if s and any(s.endswith(e) for e in POLITE_ENDINGS):
            return True
    return False


def bare(token):
    """KoreanClause.bare: the token without trailing punctuation, laughter or emoji."""
    while token and (token[-1] in BARE_NOISE or unicodedata.category(token[-1]) == "So"):
        token = token[:-1]
    return token


def names_someone_else(text, user_calls_character=False):
    return any(not user_calls_character or not m.group().startswith("님") for m in OTHER_PERSON.finditer(text))


def split_keeping(text, marks=".!?…\n"):
    out, current = [], ""
    for ch in text:
        current += ch
        if ch in marks:
            out.append(current)
            current = ""
    return out + [current]


def honorific_to_user(reply, user=""):
    """reference reply checker.honorificToUser: the word that honors the user in a 반말 reply, or None."""
    text = without_quotes(reply)
    if names_someone_else(text) or names_someone_else(user, True):
        return None
    for s in split_keeping(text):
        m = USER_HONORIFIC.search(s)
        if m:
            a = max((i for i, ch in enumerate(s[:m.start()]) if ch.isspace()), default=-1) + 1
            b = next((i for i in range(m.end(), len(s)) if s[i].isspace()), len(s))
            return bare(s[a:b])
    return None


def casual_sentence_observed(reply):
    """VisibleRegisterState.casualSentenceObserved: a sentence of a 해요체 reply ends in 반말."""
    for sentence in SENTENCE_SPLIT.split(without_quotes(reply)):
        s = trim(sentence, " ~^ㅋㅎ\"',")
        if len(s) < 2 or s == "네" or not "가" <= s[-1] <= "힣":
            continue
        if BANMAL_ENDING.search(s):
            return True
    return False


def register_problem(reply, register, user=""):
    """The app's register checks for this character: a 반말 character's polite ending or honoring of the user
    (reference reply checker.slipsFromBanmal), a 해요체 character's 반말 sentence (personaCasualRegister)."""
    if register == "banmal":
        if ends_politely(reply):
            return "존댓말 어미"
        word = honorific_to_user(reply, user)
        return f"사용자 높임 '{word}'" if word else None
    return "반말 어미" if casual_sentence_observed(reply) else None


def identity_slip(reply, human_in_world=False, admits_ai=False):
    """VisibleLocalPolicyNote.identitySlip for the package's stance."""
    text = unicodedata.normalize("NFKC", reply).casefold()
    words = PRODUCT_WORDS + DEFENSIVE + ([] if human_in_world else REAL_HUMAN)
    if any(w in text for w in words):
        return True
    if admits_ai:
        return False
    if human_in_world:
        text = re.sub(r"(ai|인공지능|챗봇)\s*(이|가)?\s*(아니|아냐)", "", text)
    return any(w in text for w in AI_WORDS)


def says_long_ago(text):
    """reference reply checker.saysLongAgo."""
    return LONG_AGO.search(text) is not None


def sounds_unsure(reply):
    """reference reply checker.soundsUnsure: not knowing, or asking back (any question mark counts, as in the app)."""
    return UNSURE.search(reply) is not None


def says_not_knowing(reply):
    """soundsUnsure without its question-mark rule: the reply itself says it does not know (for counting dialogues)."""
    return UNSURE_WORDS.search(reply) is not None


def own_name_honorific(reply, own):
    return bool(own) and re.search(re.escape(own) + r"\s?(씨|님)", reply) is not None


def calls_user_by_role(reply, user, allows_you=False):
    return (not allows_you and "당신" in reply) or ("사용자" in reply and "사용자" not in user)


def talks_about_records(reply, user):
    return re.search(r"기록|저장|데이터", user) is None and RECORDS.search(reply) is not None


def familiar_words(reply, user):
    """Kit rule: words of an old acquaintance ("오랜만", "지난번") in a first conversation. A word the user said first
    counts only when the reply goes along with it ("오랜만이다!"), not when it questions it ("오랜만이라니? 우리 처음 같은데")."""
    found = []
    for m in FAMILIAR.finditer(reply):
        questioned = re.match(r"\s*(?:이)?(?:라니|라고\?|이요\?|요\?|\?)", reply[m.end():]) is not None
        if m.group().strip() not in user or not questioned:
            found.append(m.group())
    return found


def self_intro(reply, own):
    """Kit rule: the reply names the character as an introduction ("나는 인물야", "인물예요")."""
    pattern = re.escape(own) + r"(?:이)?(?:야|예요|이에요|입니다|라고|이라고)"
    return bool(own) and re.search(pattern, reply) is not None


def foreign_words(reply, user, allowed=""):
    """Kit rule: Latin words the user did not write (MBTI types and the package's own words aside), and Chinese or
    Japanese characters."""
    seen = (user + " " + allowed).lower()
    latin = [w for w in re.findall(r"[A-Za-z]{2,}", reply) if w.lower() not in seen and not MBTI.match(w)]
    return latin + re.findall(r"[぀-ヿ一-鿿]+", reply)


def name_register(reply, user_name, register, vocative):
    """Kit rule: a 반말 character calls the user "사용자 씨" (the app rewrites it, reference addressing rule);
    a 해요체 character calls the user "사용자야"."""
    if register == "banmal":
        return re.search(re.escape(user_name) + r"\s?씨", reply) is not None
    return re.search(re.escape(vocative) + r"(?=[\s,!?~.…]|$)", reply) is not None


# ---------------------------------------------------------------------------------------------------------------
# Next-token readings
# ---------------------------------------------------------------------------------------------------------------

EOG_IDS = {1, 106}  # Gemma 4 <eos>, <turn|>
CLOSERS = set(".!?…\n~^ㅋㅎ")
HANGUL_TAIL = re.compile(r"[가-힣]+$")
HANGUL_HEAD = re.compile(r"^[가-힣]+")


def decode(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def cands_from(entry):
    """The candidates of one completion_probabilities entry: [{"text", "end", "p"}]."""
    return [{"text": "" if c["id"] in EOG_IDS else decode(bytes(c.get("bytes") or [])), "end": c["id"] in EOG_IDS,
             "p": math.exp(c["logprob"])} for c in entry.get("top_logprobs", [])]


def pack_steps(probabilities):
    """The generated tokens and their candidates in a compact, cacheable form: [[piece bytes as hex, [[text, end, p],
    ...]], ...]. Readings are computed from this at report time (unpack_steps), so a change to the reading code needs no
    new model calls."""
    packed = []
    for entry in probabilities:
        piece = b"" if entry["id"] in EOG_IDS else bytes(entry.get("bytes") or [])
        packed.append([piece.hex(), pack_cands(cands_from(entry))])
    return packed


def pack_cands(cands):
    return [[c["text"], c["end"], float(f"{c['p']:.6g}")] for c in cands]


def unpack_cands(packed):
    return [{"text": text, "end": end, "p": p} for text, end, p in packed]


def unpack_steps(packed):
    """[{"prefix", "piece", "cands": [{"text", "end", "p"}]}]. A candidate that is part of a character (a byte piece) has
    text None and is not counted."""
    steps, raw = [], b""
    for piece_hex, cands in packed:
        piece = bytes.fromhex(piece_hex)
        steps.append({"prefix": raw.decode("utf-8", "ignore"), "piece": piece.decode("utf-8", "ignore"),
                      "cands": unpack_cands(cands)})
        raw += piece
    return steps


def steps_from(probabilities):
    """unpack_steps(pack_steps(...)) for llama-server completion_probabilities."""
    return unpack_steps(pack_steps(probabilities))


def first_sentence_end(steps):
    """Index of the step that ends the reply's first sentence with a word of two or more syllables in it."""
    text = ""
    for i, step in enumerate(steps):
        text += step["piece"]
        if any(ch in step["piece"] for ch in ".!?…\n") and re.search(r"[가-힣]{2}", text):
            return i
    return len(steps) - 1


def polite_word(word):
    return len(word) >= 2 and any(word.endswith(e) for e in POLITE_ENDINGS) and not word.endswith(YO_NOUNS)


def banmal_word(word):
    """A word that ends a sentence in 반말 when nothing follows it ("좋아", "그랬구나"), as casualSentenceObserved reads a
    sentence end."""
    return len(word) >= 2 and word != "네" and not polite_word(word) and BANMAL_ENDING.search(word) is not None


def word_at(prefix, text):
    """(the word this candidate writes into, the candidate's Hangul lead, the rest of the candidate). A candidate that
    starts with a space begins a new word."""
    if text[:1].isspace():
        before, body = "", text.lstrip()
    else:
        tail = HANGUL_TAIL.search(prefix)
        before, body = (tail.group() if tail else ""), text
    head = HANGUL_HEAD.match(body)
    lead = head.group() if head else ""
    return before, lead, body[len(lead):]


def closes(cand):
    """The candidate ends the sentence here: the end of the reply, or sentence-closing punctuation or laughter."""
    return cand["end"] or (bool(cand["text"]) and cand["text"][0] in CLOSERS)


def other_register_hit(prefix, cand, register):
    """Whether this candidate itself supplies the other register here. 반말 character: it writes the polite ending
    ("하" + "세요"); a word that is already polite before it ("…안녕하세요" + ".") does not count again, so a reply that
    has slipped does not read as pulled by construction. 해요체 character: it closes a sentence on a 반말 ending, either
    after one ("좋아" + ".") or within itself ("좋" + "아.")."""
    if cand["text"] is None:
        return False
    before, lead, rest = word_at(prefix, cand["text"])
    if register == "banmal":
        return lead != "" and polite_word(before + lead) and not polite_word(before)
    if lead == "":
        return closes(cand) and banmal_word(before)
    return rest[:1] in CLOSERS and rest != "" and banmal_word(before + lead)


def branch_candidate(prefix, cand, register):
    """For a 해요체 character: a candidate that completes a 반말-ending word without closing it ("좋" + "아"). Whether the
    sentence then ends there (반말) or goes on with "요" is only seen one token later (Gemma writes "아요" as one token,
    so "좋" + "아" + "." is a two-step path); diagnose_lora asks the model for that next step."""
    if register != "haeyo" or not cand["text"] or cand["end"]:
        return False
    before, lead, rest = word_at(prefix, cand["text"])
    return lead != "" and rest == "" and banmal_word(before + lead)


BRANCH_MIN_P = 0.02  # branch candidates below this are not followed
MAX_BRANCHES = 12  # per reply, the most likely ones (nouns such as "가게" end in 게 too and take some)


def branch_points(steps, register):
    """[(step index, candidate text)] to follow one token further (branch_candidate), in the first sentence; the
    candidate the reply itself took is not asked again, since the next step shows its branch."""
    found = []
    for i, step in enumerate(steps[:first_sentence_end(steps) + 1]):
        for cand in step["cands"]:
            if cand["p"] >= BRANCH_MIN_P and cand["text"] != step["piece"] and branch_candidate(step["prefix"], cand, register):
                found.append((cand["p"], i, cand["text"]))
    return [(i, text) for _, i, text in sorted(found, reverse=True)[:MAX_BRANCHES]]


def closing_mass(cands):
    return sum(c["p"] for c in cands if c["text"] is not None and closes(c))


def register_pull(steps, register, branches=None):
    """The largest probability mass on the other register at any point of the first sentence, and where it was.
    `branches` ({"<step>|<text>": next-token candidates}) carries the 해요체 two-step paths: the mass there is
    P(stem candidate) × P(sentence end after it). The candidate the reply took uses the next step's candidates."""
    branches = branches or {}
    best = {"mass": 0.0, "prefix": "", "piece": ""}
    for i, step in enumerate(steps[:first_sentence_end(steps) + 1]):
        hits = [(c["p"], c["text"] or "(끝)") for c in step["cands"] if other_register_hit(step["prefix"], c, register)]
        for c in step["cands"]:
            if not branch_candidate(step["prefix"], c, register):
                continue
            after = (steps[i + 1]["cands"] if c["text"] == step["piece"] and i + 1 < len(steps)
                     else branches.get(f"{i}|{c['text']}"))
            if after:
                hits.append((c["p"] * closing_mass(after), c["text"] + "(끝)"))
        mass = sum(p for p, _ in hits)
        if mass > best["mass"]:
            best = {"mass": mass, "prefix": step["prefix"][-14:], "piece": max(hits)[1]}
    return best


def onset_pull(steps, starts, inside):
    """The largest mass on candidates that start a run (`starts(text)`) where the reply is not already in one
    (`inside(prefix)`), over the whole reply."""
    best = {"mass": 0.0, "prefix": "", "piece": ""}
    for step in steps:
        if inside(step["prefix"]):
            continue
        hits = [c for c in step["cands"] if c["text"] and starts(c["text"])]
        mass = sum(c["p"] for c in hits)
        if mass > best["mass"]:
            best = {"mass": mass, "prefix": step["prefix"][-14:], "piece": max(hits, key=lambda c: c["p"])["text"]}
    return best


def latin_pull(steps):
    return onset_pull(steps, lambda t: re.match(r"\s?[A-Za-z]", t) is not None, lambda p: re.search(r"[A-Za-z]$", p) is not None)


def laugh_pull(steps):
    return onset_pull(steps, lambda t: re.match(r"\s?[ㅋㅎ]", t) is not None, lambda p: re.search(r"[ㅋㅎ]$", p) is not None)


def consistent(cand, remaining, at_start):
    text = cand["text"].lstrip() if at_start and cand["text"] else cand["text"]
    return bool(text) and not cand["end"] and (remaining.startswith(text) or text.startswith(remaining))


def next_opening_step(cands, remaining, at_start):
    """The most likely candidate on the way to `remaining`, or None when none of the top candidates is."""
    fits = [c for c in cands if consistent(c, remaining, at_start)]
    return max(fits, key=lambda c: c["p"]) if fits else None
