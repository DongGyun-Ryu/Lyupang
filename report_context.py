"""Source-only, bounded period/basis references. Financial values are never inputs."""
from __future__ import annotations

import hashlib
import re

from check_facts import require
from evidence_context import sentence_spans
from report_claims import RELATED, OTHER_NAMED_SUBJECT

POLICY = "source_period_basis_context_v1"
BASIS = re.compile(r"(?:연결|별도)\s*기준|(?:연결|별도)\s*재무제표(?:의\s*데이터)?(?:를|에)?\s*(?:기준|근거)")
YEAR = re.compile(r"(?<![0-9])(?P<year>(?:19|20)[0-9]{2})년")
ORDINAL = re.compile(r"제\s*(?P<ordinal>[0-9]{1,3})기")
YEAR_ORDINAL = re.compile(r"(?P<year>(?:19|20)[0-9]{2})년\s*\(\s*제\s*(?P<ordinal>[0-9]{1,3})기\s*\)")
SUBPERIOD = re.compile(r"분기|상반기|하반기|[0-9]+\s*개월|[0-9]+\s*월|말\s*현재")
FORWARD = re.compile(r"예상|전망|계획|목표|예정|향후|내년|작성할|적용할")
REFERENCE = re.compile(r"참고|참조|인용")
METRIC = re.compile(r"매출|영업수익|영업이익|순이익|영업실적|재무상태")


def basis_value(text):
    return "CFS" if text.startswith("연결") else "OFS"


def source_span(block, start, end, method, field, value, claim):
    return {"code": claim["code"], "receipt_no": claim["receipt_no"], "block_id": block["id"],
        "block_sha256": block["text_sha256"], "start": start, "end": end, "text": block["text"][start:end],
        "headings": block["headings"], "method": method, "field": field, "value": value}


class ContextIndex:
    def __init__(self, blocks, company_name, *, code=None, receipt=None):
        self.blocks = list(blocks.values()) if isinstance(blocks, dict) else list(blocks)
        self.code, self.receipt = code, receipt
        require(len({b["id"] for b in self.blocks}) == len(self.blocks))
        for b in self.blocks:
            require(hashlib.sha256(b["text"].encode()).hexdigest() == b["text_sha256"], "문맥 근거의 본문 해시가 다릅니다.")
        self.by_id = {b["id"]: b for b in self.blocks}
        self.positions = {b["id"]: i for i, b in enumerate(self.blocks)}
        self.ordinals = {}
        aliases = [company_name, "당사"] + (["네이버"] if company_name == "NAVER" else [])
        self.aliases = {a.casefold() for a in aliases}
        own = re.compile("|".join(re.escape(a) for a in aliases if a))
        for block in self.blocks:
            if block["kind"] != "paragraph": continue
            for match in YEAR_ORDINAL.finditer(block["text"]):
                left, right = max(0, match.start()-25), min(len(block["text"]), match.end()+30)
                around = block["text"][left:right]
                if own.search(around) and not self.foreign(around) and not SUBPERIOD.search(around) and not FORWARD.search(around):
                    self.ordinals.setdefault(int(match["ordinal"]), []).append((block, match.start(), match.end(), int(match["year"])))

    def foreign(self, text):
        return bool(RELATED.search(text) or any(m.group().casefold() not in self.aliases
                    and not re.search(r"당사|연결|우리", m.group()) for m in OTHER_NAMED_SUBJECT.finditer(text)))

    def resolve(self, claim, quantity):
        require((self.code is None or claim["code"] == self.code) and (self.receipt is None or claim["receipt_no"] == self.receipt), "다른 종목·접수번호의 문맥은 연결하지 않습니다.")
        block = self.by_id[claim["block_id"]]
        require(claim["block_sha256"] == block["text_sha256"] and claim["headings"] == block["headings"])
        require(block["text"][claim["start"]:claim["end"]] == claim["text"], "문맥 대상의 원문 위치가 다릅니다.")
        require(claim["text"][quantity["start"]:quantity["end"]] == quantity["text"])
        direct_years = [p for p in claim["period_expressions"] if p["kind"] == "explicit_year"]
        year = direct_years[0]["year"] if len(direct_years) == 1 and re.fullmatch(r"[0-9]{4}년", direct_years[0]["text"]) else None
        basis_tokens = list(BASIS.finditer(claim["text"]))
        basis_values = {basis_value(m.group()) for m in basis_tokens}
        scope = next(iter(basis_values)) if len(basis_values) == 1 else None
        result = {"policy": POLICY, "claim_id": claim["id"], "quantity_start": quantity["start"], "quantity_end": quantity["end"],
            "year": year, "scope": scope, "year_method": "sentence_explicit" if year else "unresolved",
            "basis_method": "sentence_explicit" if scope else "unresolved", "evidence": [], "conflicts": []}
        blocked = set()
        if scope is not None and not claim["basis_expressions"]:
            for token in basis_tokens:
                result["evidence"].append(source_span(block, claim["start"]+token.start(), claim["start"]+token.end(),
                    "sentence_explicit", "scope", scope, claim))
        # A mixed enumeration can be divided only at its own explicit basis cues.
        if len(basis_values) > 1:
            preceding = [m for m in basis_tokens if m.end() <= quantity["start"]]
            unsafe = re.search(r"아니|아닌|않|제외|비교|대신|가정|참고|참조|인용", claim["text"])
            if preceding and not unsafe:
                chosen = preceding[-1]
                result.update(scope=basis_value(chosen.group()), basis_method="clause_explicit")
                result["evidence"].append(source_span(block, claim["start"]+chosen.start(), claim["start"]+chosen.end(),
                    "clause_explicit", "scope", result["scope"], claim))
            else: result["conflicts"].append("basis_mixed_before_quantity")
        candidates = {"year": [], "scope": []}
        position = self.positions[block["id"]]

        def consider(anchor, start, end, method):
            text = anchor["text"][start:end]
            if FORWARD.search(text) or REFERENCE.search(text) or self.foreign(text) or SUBPERIOD.search(text): return
            years = {int(m["year"]) for m in YEAR.finditer(text)}
            scopes = {basis_value(m.group()) for m in BASIS.finditer(text)}
            if len(years) > 1 or len(scopes) > 1:
                if len(years) > 1: blocked.add("year")
                if len(scopes) > 1: blocked.add("scope")
                result["conflicts"].append("mixed_context_values"); return
            # Cross-field contradictions invalidate borrowing, never explicit text.
            if year is not None and years and years != {year}: return
            if result["scope"] is not None and scopes and scopes != {result["scope"]}: return
            if len(years) == 1:
                candidates["year"].append(source_span(anchor, start, end, method, "year", next(iter(years)), claim))
            if len(scopes) == 1:
                candidates["scope"].append(source_span(anchor, start, end, method, "scope", next(iter(scopes)), claim))

        # The paragraph must be uniformly scoped, without another entity, quote,
        # forecast or interim period; another sentence is not a global default.
        paragraph = block["text"]
        if len({basis_value(m.group()) for m in BASIS.finditer(paragraph)}) > 1: blocked.add("scope")
        if len({int(m["year"]) for m in YEAR.finditer(paragraph)}) > 1: blocked.add("year")
        if self.foreign(paragraph) or REFERENCE.search(paragraph): blocked.update(("year", "scope"))
        if (len({basis_value(m.group()) for m in BASIS.finditer(paragraph)}) <= 1
                and len({int(m["year"]) for m in YEAR.finditer(paragraph)}) <= 1
                and not (self.foreign(paragraph) or FORWARD.search(paragraph) or REFERENCE.search(paragraph) or SUBPERIOD.search(paragraph))):
            for start, end in sentence_spans(paragraph):
                if end <= claim["start"] or start >= claim["end"]:
                    if METRIC.search(paragraph[start:end]): consider(block, start, end, "same_paragraph")

        # Only the active heading path, never another chapter's financial heading.
        for heading in self.blocks[max(0, position-80):position]:
            if heading["kind"] == "heading" and METRIC.search(heading["text"]) and heading["headings"] == claim["headings"][:len(heading["headings"])]:
                consider(heading, 0, len(heading["text"]), "active_heading")

        # Adjacent declarations must explicitly say that they cover below/above.
        for offset in (-1, 1):
            i = position+offset
            if not 0 <= i < len(self.blocks): continue
            anchor = self.blocks[i]
            if anchor["kind"] != "paragraph" or anchor["headings"] != block["headings"]: continue
            for start, end in sentence_spans(anchor["text"]):
                text = anchor["text"][start:end]
                direction = r"아래|이하|다음" if offset < 0 else r"상기|위의|앞서"
                if re.match(r"^(?:※\s*)?(?:"+direction+r")", text) and not re.search(r"표|주석", text.replace("재무제표", "")) and re.search(r"기준|작성|적용|실적", text):
                    consider(anchor, start, end, "adjacent_declaration")

        # A report-specific chapter declaration is recognized by its exact scope,
        # even if the same introductory paragraph discusses other statements.
        for anchor in self.blocks[max(0, position-80):position]:
            if anchor["kind"] != "paragraph": continue
            chapter = next((h for h in anchor["headings"] if "경영진단" in h), None)
            if chapter is None or chapter not in claim["headings"]: continue
            for start, end in sentence_spans(anchor["text"]):
                text = anchor["text"][start:end]
                compact = re.sub(r"[\s'\"‘’“”]", "", text)
                if re.search(r"본.*이사의경영진단및분석의견.*은.*재무제표.*기준.*작성", compact):
                    consider(anchor, start, end, "explicit_chapter_declaration")

        # A fiscal ordinal is mapped only by an explicit same-company source pair.
        ordinals = {int(m["ordinal"]) for m in ORDINAL.finditer(claim["text"])}
        if not direct_years and len(ordinals) == 1:
            for anchor, start, end, value in self.ordinals.get(next(iter(ordinals)), []):
                candidates["year"].append(source_span(anchor, start, end, "explicit_fiscal_year_pair", "year", value, claim))

        for field in ("year", "scope"):
            if result[field] is not None: continue
            if field in blocked:
                result["conflicts"].append(field+"_context_conflict"); continue
            # Explicit contradictory/multiple values in the target cannot be fixed
            # by selecting a nearby declaration or a value that fits the finances.
            if field == "year" and direct_years: continue
            if field == "scope" and basis_tokens: continue
            values = {c["value"] for c in candidates[field]}
            if len(values) > 1:
                result["conflicts"].append(field+"_context_conflict"); continue
            if len(values) == 1:
                result[field] = next(iter(values))
                result["year_method" if field == "year" else "basis_method"] = "source_context"
                result["evidence"].extend(candidates[field])
        # Even an explicit fiscal mapping cannot make an interim claim annual.
        if SUBPERIOD.search(claim["text"]):
            result["year"] = None; result["year_method"] = "unresolved"
            result["evidence"] = [e for e in result["evidence"] if e["field"] != "year"]
        same = 0
        for other in claim["measurements"]:
            if other["metric"] != quantity["metric"] or other["unit"] != "KRW": continue
            other_scope = next(iter(basis_values)) if len(basis_values) == 1 else None
            if len(basis_values) > 1:
                preceding = [m for m in basis_tokens if m.end() <= other["start"]]
                if preceding: other_scope = basis_value(preceding[-1].group())
            if other_scope is None and not basis_tokens: other_scope = result["scope"]
            same += other_scope == result["scope"]
        result["same_metric_scope_amounts"] = same
        return result
