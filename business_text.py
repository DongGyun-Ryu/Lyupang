"""Source-linked extraction of company statements, never causal market conclusions."""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from dart_client import LIMIT
from lyupang import CollectionError, SEOUL, parse_date

MAX_XML = 64 * 1024 * 1024
MAX_EXPANDED = 128 * 1024 * 1024
MAX_NODES = 500_000
CATEGORIES = {"overview": "사업 개요", "products": "주요 제품·서비스",
              "performance": "실적 변화에 관한 설명", "risks": "위험요인"}
MAX_QUOTES = 5
QUOTE_LENGTH = 1200
IGNORED = {"SCRIPT", "STYLE", "NOSCRIPT"}
# DART bodies contain literal R&D and < TV ... > / <이사 ...> labels.
# Preserve XML references and literal sections; never recover broken structure.
PROTECTED_XML = re.compile(r"<!\[CDATA\[.*?\]\]>|<!--.*?-->|<\?.*?\?>", re.S)
BARE_AMPERSAND = re.compile(r"&(?![^&<>\s;]+;|#)")
LITERAL_LT = re.compile(r"<(?=[\s0-9=])|<[가-힣][^<>=/'\"\r\n]{0,160}[ \t][^\s<>=/'\"][^<>=/'\"\r\n]{0,160}>")
MAX_LITERAL_CHARACTERS = 100_000


def local(tag):
    return tag.rsplit("}", 1)[-1].upper()


def compact(text):
    return re.sub(r"\s+", "", text)


def text_of(node):
    parts = []

    def visit(element):
        if local(element.tag) in IGNORED:
            return
        if element.text:
            parts.append(element.text)
        for child in element:
            if local(child.tag) == "BR":
                parts.append("\n")
            else:
                visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(node)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def prepare_xml(raw):
    """Encode observed literal characters for parsing without rewriting source files.

    CDATA, comments and processing instructions are copied byte-for-byte as text.
    Unknown entity references and mismatched tags remain errors. Korean angle
    labels must contain whitespace and no XML attribute/closing-tag syntax;
    legitimate Unicode element names and attributes are left alone.
    """
    if len(raw) > MAX_XML:
        raise CollectionError("사업보고서 XML이 지원 크기를 넘었습니다.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise CollectionError("사업보고서 XML이 지원하는 UTF-8 형식이 아닙니다.") from None
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.I):
        raise CollectionError("사업보고서 XML에 지원하지 않는 DTD·엔티티 선언이 있습니다.")
    counts = {"literal_ampersands_encoded": 0, "literal_less_than_encoded": 0}

    def replace(match, key, replacement):
        counts[key] += 1
        if sum(counts.values()) > MAX_LITERAL_CHARACTERS:
            raise CollectionError("사업보고서의 특수문자 표기 수가 지원 범위를 넘었습니다.")
        return replacement + match.group()[1:]

    def encode(segment):
        segment = BARE_AMPERSAND.sub(lambda m: replace(m, "literal_ampersands_encoded", "&amp;"), segment)
        return LITERAL_LT.sub(lambda m: replace(m, "literal_less_than_encoded", "&lt;"), segment)

    parts, end = [], 0
    for protected in PROTECTED_XML.finditer(text):
        parts.append(encode(text[end:protected.start()]))
        parts.append(protected.group())
        end = protected.end()
    parts.append(encode(text[end:]))
    prepared = "".join(parts)
    prepared_bytes = prepared.encode("utf-8")
    if len(prepared_bytes) > MAX_XML:
        raise CollectionError("특수문자 표기를 처리한 사업보고서가 지원 크기를 넘었습니다.")
    return prepared, {"policy": "literal_text_characters_v1", **counts,
                      "normalization_applied": bool(sum(counts.values())),
                      "original_xml_sha256": hashlib.sha256(raw).hexdigest(),
                      "prepared_xml_sha256": hashlib.sha256(prepared_bytes).hexdigest()}


def validated_root(raw):
    text, _ = prepare_xml(raw)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        line, column = exc.position
        raise CollectionError(f"사업보고서 XML을 읽지 못했습니다. 원문 형식을 확인해야 합니다. "
                              f"(처리 본문 {line}행 {column}열, 형식 코드 {exc.code})") from None
    except ValueError:
        raise CollectionError("사업보고서 XML의 선언 형식을 확인해야 합니다.") from None
    stack, count = [(root, 0)], 0
    while stack:
        element, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > 80:
            raise CollectionError("사업보고서 XML의 노드 수 또는 중첩이 지원 범위를 넘었습니다.")
        stack.extend((child, depth + 1) for child in element)
    return root


def annual_identity(root, corp):
    if local(root.tag) != "DOCUMENT":
        return False
    names = [e for e in root.iter() if local(e.tag) == "DOCUMENT-NAME"]
    if len(names) != 1:
        return False
    name = names[0]
    title = compact(re.sub(r"^(?:\[[^\]]+\]\s*)*", "", text_of(name)))
    if title != "사업보고서" or (name.get("ACODE") and name.get("ACODE") != "11011"):
        return False
    companies = [e for e in root.iter() if local(e.tag) == "COMPANY-NAME" and e.get("AREGCIK")]
    return bool(companies) and {e.get("AREGCIK").strip() for e in companies} == {corp}


def main_document(payload, receipt, corp):
    """Prefer the receipt-named main file; never replace its failure with an attachment."""
    if len(payload) > LIMIT:
        raise CollectionError("사업보고서 압축 파일이 지원 크기를 넘었습니다.")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            entries = bundle.infolist()
            if len(entries) > 200 or sum(e.file_size for e in entries) > MAX_EXPANDED:
                raise CollectionError("사업보고서 압축 파일의 항목 수 또는 전체 크기가 지원 범위를 넘었습니다.")
            names = set()
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename
                        or entry.filename.casefold() in names or entry.flag_bits & 1
                        or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                    raise CollectionError("사업보고서 압축 파일에 지원하지 않는 경로·중복·암호화 항목이 있습니다.")
                names.add(entry.filename.casefold())
            xmls = [e for e in entries if not e.is_dir() and e.filename.lower().endswith(".xml")]
            preferred = [e for e in xmls if PurePosixPath(e.filename).name.casefold() == receipt + ".xml"]
            if len(preferred) > 1:
                raise CollectionError("접수번호에 해당하는 본문 파일이 여러 개입니다.")
            selected = []
            for entry in preferred or xmls:
                if entry.file_size > MAX_XML:
                    raise CollectionError("사업보고서 XML이 지원 크기를 넘었습니다.")
                with bundle.open(entry) as stream:
                    raw = stream.read(MAX_XML + 1)
                try:
                    root = validated_root(raw)
                except CollectionError:
                    if preferred:
                        raise
                    continue  # Non-main attachments cannot stand in for an annual report.
                if annual_identity(root, corp):
                    selected.append((entry.filename, raw, root))
                elif preferred:
                    raise CollectionError("접수번호 본문의 기업번호 또는 사업보고서 문서 종류를 확인하지 못했습니다.")
            if len(selected) != 1:
                raise CollectionError("기업번호가 일치하는 사업보고서 본문을 하나로 특정하지 못했습니다.")
            return selected[0]
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError):
        raise CollectionError("사업보고서 압축 파일을 읽지 못했습니다.") from None


def extract_blocks(root):
    blocks = []

    def add(node, headings, path, kind, text=None):
        content = text if text is not None else text_of(node)
        if content:
            blocks.append({"id": f"b{len(blocks)+1:06d}", "kind": kind, "headings": list(headings),
                           "xml_path": path, "text": content,
                           "text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})

    def walk(node, headings, path):
        tag = local(node.tag)
        if tag in IGNORED:
            return
        if tag == "TABLE":
            rows = []
            for row in node.iter():
                if local(row.tag) != "TR":
                    continue
                cells = [text_of(c) for c in row if local(c.tag) in {"TD", "TH", "TE", "TU"}]
                if cells:
                    rows.append(" | ".join(cells))
            add(node, headings, path, "table", "\n".join(rows) if rows else text_of(node))
            return
        if tag in {"P", "PARAGRAPH"}:
            add(node, headings, path, "paragraph")
            return
        if tag == "TITLE":
            add(node, headings, path, "heading")
            return
        scoped = tag.startswith("SECTION-")
        title = next((text_of(c) for c in node if local(c.tag) == "TITLE"), None) if scoped else None
        current = headings + (title,) if title else headings
        for index, child in enumerate(node, 1):
            if local(child.tag) == "TITLE" and not scoped:
                name = text_of(child)
                if re.match(r"^(?:[IVX]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)[.．\s]", name):
                    current = (name,)
                else:
                    current = headings + (name,)
            walk(child, current, f"{path}/{local(child.tag)}[{index}]")

    walk(root, (), "/DOCUMENT[1]")
    if not blocks:
        raise CollectionError("지원하는 제목·문단·표 구조에서 사업보고서 본문을 추출하지 못했습니다.")
    return blocks


def extract_evidence(blocks):
    candidates = {key: [] for key in CATEGORIES}
    for block in blocks:
        if block["kind"] == "heading" or len(block["text"]) < 25:
            continue
        headings = compact(" / ".join(block["headings"]))
        in_business = "사업의내용" in headings or "사업개요" in headings or "사업의개요" in headings
        in_management = "경영진단" in headings or "경영성과" in headings or "영업실적" in headings
        if not (in_business or in_management):
            continue
        text = block["text"]
        categories = []
        if in_business and ("사업의개요" in headings or "사업개요" in headings):
            categories.append("overview")
        if in_business and any(word in headings for word in ("주요제품", "제품및서비스", "주요서비스", "매출및수주")):
            categories.append("products")
        if (block["kind"] == "paragraph" and re.search(r"매출|영업이익|순이익|판매량|수익성|실적", text)
                and re.search(r"증가|감소|상승|하락|개선|악화|변동|성장", text)):
            categories.append("performance")
        if re.search(r"위험|리스크|불확실|경쟁|환율", headings + text):
            categories.append("risks")
        for category in categories:
            start = 0
            # Preserve an exact substring; no generated explanation or numeric inference.
            if category == "performance":
                match = re.search(r"매출|영업이익|순이익|판매량|수익성|실적", text)
                start = max(0, match.start() - 200)
            elif category == "risks":
                match = re.search(r"위험|리스크|불확실|경쟁|환율", text)
                start = max(0, match.start() - 200) if match else 0
            quote = text[start:start + QUOTE_LENGTH]
            # A long paragraph can match outside the displayed window. Do not label
            # its excerpt as performance evidence unless that window also matches.
            if category == "performance" and not re.search(r"증가|감소|상승|하락|개선|악화|변동|성장", quote):
                continue
            candidates[category].append({"block_id": block["id"], "block_text_sha256": block["text_sha256"],
                "headings": block["headings"], "xml_path": block["xml_path"], "kind": block["kind"],
                "quote": quote, "start": start, "end": start + len(quote),
                "is_partial_block": start != 0 or len(quote) != len(text),
                "forward_looking_language": bool(re.search(r"전망|예상|계획|예측|기대|목표|추정", quote)),
                "reason_language_found": bool(re.search(r"영향|기인|때문|요인|인해|힘입|원인|덕분|부진", quote))})
    return {key: {"candidate_count": len(rows), "shown_count": min(len(rows), MAX_QUOTES),
                  "quotes": rows[:MAX_QUOTES]} for key, rows in candidates.items()}


def evidence_as_of(events, as_of):
    eligible = [e for e in events if parse_date(e["receipt_date"]) < as_of and parse_date(e["period_end"]) < as_of]
    if not eligible:
        return {"status": "unavailable", "reason": "no_previous_annual_report"}
    period = max(e["period_end"] for e in eligible)
    latest = max(e["receipt_date"] for e in eligible if e["period_end"] == period)
    selected = [e for e in eligible if e["period_end"] == period and e["receipt_date"] == latest]
    if len(selected) != 1:
        return {"status": "unavailable", "reason": "ambiguous_same_day_reports"}
    event = selected[0]
    if event["status"] != "read":
        return {"status": "unavailable", "reason": "latest_report_unreadable", "receipt_no": event["receipt_no"]}
    return {"status": "available", "receipt_no": event["receipt_no"], "event": event}


def cached_document(cache, code, corp, receipt, client):
    if not re.fullmatch(r"[0-9]{6}", code) or not re.fullmatch(r"[0-9]{8}", corp) or not re.fullmatch(r"[0-9]{14}", receipt):
        raise CollectionError("사업보고서 원문 저장용 식별번호가 올바르지 않습니다.")
    cache.mkdir(parents=True, exist_ok=True)
    for folder in sorted(cache.glob(f"{code}_{receipt}_*"), reverse=True):
        if not folder.is_dir() or folder.is_symlink():
            continue
        try:
            meta = json.loads((folder / "source.json").read_text(encoding="utf-8"))
            raw = (folder / "original.zip").read_bytes()
            if (meta.get("endpoint") == "document.xml" and meta.get("code") == code and meta.get("corp_code") == corp
                    and meta.get("receipt_no") == receipt and meta.get("sha256") == hashlib.sha256(raw).hexdigest()
                    and len(raw) <= LIMIT and zipfile.is_zipfile(io.BytesIO(raw))):
                return raw, meta, folder
        except (OSError, ValueError):
            continue
    raw = client.original_document(receipt)
    if len(raw) > LIMIT or not zipfile.is_zipfile(io.BytesIO(raw)):
        raise CollectionError("사업보고서 본문 압축 파일을 받지 못해 저장하지 않았습니다.")
    meta = {"provider": "OpenDART", "endpoint": "document.xml", "code": code, "corp_code": corp,
            "receipt_no": receipt, "sha256": hashlib.sha256(raw).hexdigest(), "retrieved_at": datetime.now(SEOUL).isoformat()}
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=cache))
    folder = cache / f"{code}_{receipt}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "original.zip").write_bytes(raw)
        (staging / "source.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staging, folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return raw, meta, folder


def source_html(blocks, title):
    esc = lambda v: html.escape(str(v))
    rows = "".join(f"<section id='{b['id']}'><small>{esc(b['id'])} · {esc(' / '.join(b['headings']))}</small>"
                   f"<pre>{esc(b['text'])}</pre></section>" for b in blocks)
    return f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>{esc(title)}</title>
<style>body{{max-width:1000px;margin:30px auto;padding:0 20px;font-family:system-ui;line-height:1.8}}pre{{white-space:pre-wrap;font-family:inherit}}section{{padding:15px;border-bottom:1px solid #ddd}}:target{{background:#fff2b5}}</style>
<h1>{esc(title)}</h1><p>보고서 XML에서 추출한 평문입니다. 문단·표와 제목 경로를 보존합니다. 표의 병합셀·단위 관계를 자동 해석하거나 수치를 재계산하지 않았습니다.</p>{rows}</html>"""
