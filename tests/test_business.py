import hashlib
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import business
import business_text as bt
from dart_client import DartError
from lyupang import CollectionError
from test_dart import client_with

CORP = "00126380"
RECEIPT = "20250311000001"


def xml_document(corp=CORP, title="사업보고서", *, extra=""):
    return f'''<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><DOCUMENT-NAME ACODE="11011">{title}</DOCUMENT-NAME>
<DOCUMENT-HEADER><COMPANY-NAME AREGCIK="{corp}">예시회사</COMPANY-NAME></DOCUMENT-HEADER><BODY>
<SECTION-1><TITLE>II. 사업의 내용</TITLE>
<SECTION-2><TITLE>1. 사업의 개요</TITLE><P>당사는 반도체를 설계하고 제조하여 국내외 고객에게 판매하는 사업을 영위하고 있습니다.</P></SECTION-2>
<SECTION-2><TITLE>2. 주요 제품 및 서비스</TITLE><TABLE><TR><TH>사업부문</TH><TH>주요제품 및 서비스</TH><TH>표시단위</TH></TR>
<TR><TD>제조 부문</TD><TD>전자부품과 장비의 생산 및 판매</TD><TD>백만원</TD></TR></TABLE></SECTION-2>
<SECTION-2><TITLE>3. 위험관리</TITLE><P>해외 시장의 수요 불확실성과 환율 변동은 당사의 영업 활동에 영향을 줄 수 있습니다.</P></SECTION-2>
</SECTION-1><SECTION-1><TITLE>IV. 이사의 경영진단 및 분석의견</TITLE>
<SECTION-2><TITLE>1. 영업실적</TITLE><P>매출액은 신규 제품의 판매 증가에 힘입어 전년 대비 증가하였습니다. 향후 실적 개선을 예상하고 있습니다.</P>
{extra}</SECTION-2></SECTION-1></BODY></DOCUMENT>'''.encode("utf-8")


def archive(xml=None, receipt=RECEIPT, extras=None):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(receipt + ".xml", xml if xml is not None else xml_document())
        for name, content in (extras or {}).items():
            z.writestr(name, content)
    return raw.getvalue()


def filing(receipt=RECEIPT, day="2025-03-11", title="사업보고서 (2024.12)"):
    return {"receipt_no": receipt, "receipt_date": day, "report_name": title}


class BusinessTextTests(unittest.TestCase):
    def test_observed_dart_literal_characters_preserve_text_without_structure_recovery(self):
        extra = '<P>R&D 및 S&P 관련 설명입니다. < TV 시장점유율 추이 >와 <이사ㆍ감사 전체의 보수현황>을 확인합니다.</P>'
        raw = xml_document(extra=extra)
        with self.assertRaises(ET.ParseError):
            ET.fromstring(raw)
        root = bt.validated_root(raw)
        blocks = bt.extract_blocks(root)
        paragraph = next(b for b in blocks if 'R&D 및 S&P' in b['text'])
        self.assertIn('< TV 시장점유율 추이 >', paragraph['text'])
        self.assertIn('<이사ㆍ감사 전체의 보수현황>', paragraph['text'])
        prepared, metadata = bt.prepare_xml(raw)
        self.assertEqual(metadata['literal_ampersands_encoded'], 2)
        self.assertEqual(metadata['literal_less_than_encoded'], 2)
        self.assertEqual(metadata['original_xml_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(metadata['prepared_xml_sha256'], hashlib.sha256(prepared.encode()).hexdigest())
        self.assertTrue(metadata['normalization_applied'])
        prepared_twice, repeated = bt.prepare_xml(prepared.encode())
        self.assertEqual(prepared, prepared_twice)
        self.assertFalse(repeated['normalization_applied'])

    def test_cdata_comments_processing_instructions_and_existing_references_are_unchanged(self):
        extra = ('<!-- R&D < TV label > --><P><![CDATA[R&D < TV label > &amp; raw]]></P>'
                 '<?note text="R&D < literal >"?><P>R&amp;D &lt;표시&gt; &#38; &#x26;</P>')
        raw = xml_document(extra=extra)
        prepared, metadata = bt.prepare_xml(raw)
        self.assertEqual(prepared.encode(), raw)
        self.assertFalse(metadata['normalization_applied'])
        text = bt.text_of(bt.validated_root(raw))
        self.assertIn('R&D < TV label > &amp; raw', text)
        self.assertIn('R&D <표시> & &', text)

    def test_valid_unicode_elements_and_attributes_are_not_reinterpreted_as_text(self):
        raw = '<DOCUMENT><한글 설명="정상">내용</한글><한글 >내용</한글><P>원래 &amp; 표기</P></DOCUMENT>'.encode()
        prepared, metadata = bt.prepare_xml(raw)
        self.assertEqual(prepared.encode(), raw)
        self.assertEqual(bt.validated_root(raw).find('한글').text, '내용')
        self.assertFalse(metadata['normalization_applied'])

    def test_unknown_entities_mismatched_tags_and_invalid_numeric_references_still_fail(self):
        for raw in (b'<DOCUMENT><P>&unknown;</P></DOCUMENT>', b'<DOCUMENT><P>R&D</DOCUMENT>',
                    b'<DOCUMENT><P>&#0;</P></DOCUMENT>', b'<DOCUMENT><P>&#xZZ;</P></DOCUMENT>'):
            with self.subTest(raw=raw), self.assertRaisesRegex(CollectionError, '형식 코드'):
                bt.validated_root(raw)
        with patch('business_text.MAX_LITERAL_CHARACTERS', 1), self.assertRaises(CollectionError):
            bt.validated_root(b'<DOCUMENT><P>R&D S&P</P></DOCUMENT>')

    def test_observed_unpaired_labels_english_phrases_and_arrow_preserve_original_text(self):
        extra = '<P><기후변화> 설명 <당기말> 금액 <Connected Work> 제도 <Advanced Tech & SDV> 송<->수신</P>'
        raw = xml_document(extra=extra)
        with self.assertRaises(ET.ParseError):
            ET.fromstring(raw)
        root = bt.validated_root(raw)
        text = bt.text_of(root)
        for literal in ('<기후변화>', '<당기말>', '<Connected Work>', '<Advanced Tech & SDV>', '송<->수신'):
            self.assertIn(literal, text)
        prepared, metadata = bt.prepare_xml(raw)
        self.assertEqual(metadata['literal_labels_encoded'], {
            '기후변화': 1, '당기말': 1, 'Connected Work': 1, 'Advanced Tech & SDV': 1})
        self.assertEqual(metadata['literal_less_than_encoded'], 5)
        self.assertEqual(prepared, bt.prepare_xml(prepared.encode())[0])

    def test_known_paired_unicode_tags_remain_elements_and_ambiguous_or_unknown_tags_fail(self):
        raw = '<DOCUMENT><당기말>값</당기말><기후변화 설명="정상">설명</기후변화></DOCUMENT>'.encode()
        prepared, metadata = bt.prepare_xml(raw)
        self.assertEqual(prepared.encode(), raw)
        self.assertFalse(metadata['normalization_applied'])
        self.assertEqual(bt.validated_root(raw).find('당기말').text, '값')
        for raw in ('<DOCUMENT><P><미확인표제></P></DOCUMENT>',
                    '<DOCUMENT><당기말>값</당기말><P><당기말></P></DOCUMENT>'):
            with self.subTest(raw=raw), self.assertRaises(CollectionError):
                bt.validated_root(raw.encode())

    def test_observed_eng_quote_errors_preserve_attribute_characters_and_body_text(self):
        for value, expected in (('"Snow Corporation', '"Snow Corporation'),
                                ('Other receivables and others"', 'Other receivables and others"')):
            raw = f'<DOCUMENT><TE ENG="{value}" VALIGN="MIDDLE">회사 설명</TE></DOCUMENT>'.encode()
            with self.assertRaises(ET.ParseError):
                ET.fromstring(raw)
            root = bt.validated_root(raw)
            self.assertEqual(root.find('TE').get('ENG'), expected)
            self.assertEqual(root.find('TE').get('VALIGN'), 'MIDDLE')
            self.assertEqual(root.find('TE').text, '회사 설명')
            prepared, metadata = bt.prepare_xml(raw)
            self.assertEqual(metadata['eng_attributes_encoded'], 1)
            self.assertEqual(metadata['attribute_quote_characters_encoded'], 1)
            self.assertEqual(prepared, bt.prepare_xml(prepared.encode())[0])

    def test_literal_label_and_eng_processing_leave_protected_sections_and_valid_attributes_unchanged(self):
        raw = ('<DOCUMENT><!-- <당기말> ENG=""Snow" -->'
               '<?note value="<당기말> ENG=""Snow""?>'
               '<P><![CDATA[<당기말> ENG=""Snow" </당기말>]]></P>'
               '<TE ENG="&quot;Snow Corporation" VALIGN="MIDDLE">값</TE>'
               '<P>ENG=""Snow"</P></DOCUMENT>').encode()
        prepared, metadata = bt.prepare_xml(raw)
        self.assertEqual(prepared.encode(), raw)
        self.assertFalse(metadata['normalization_applied'])
        bt.validated_root(raw)
        raw = '<DOCUMENT><P><당기말><![CDATA[</당기말>]]></P></DOCUMENT>'.encode()
        self.assertIn('<당기말></당기말>', bt.text_of(bt.validated_root(raw)))

    def test_non_eng_quote_errors_and_unterminated_attributes_still_fail_and_budget_applies(self):
        for raw in (b'<DOCUMENT><TE TITLE=""Snow">body</TE></DOCUMENT>',
                    b'<DOCUMENT><TE ENG="Snow>body</TE></DOCUMENT>'):
            with self.subTest(raw=raw), self.assertRaises(CollectionError):
                bt.validated_root(raw)
        with patch('business_text.MAX_LITERAL_CHARACTERS', 1), self.assertRaises(CollectionError):
            bt.validated_root(b'<DOCUMENT><TE ENG=""Snow"">body</TE></DOCUMENT>')

    def test_receipt_identity_sections_and_every_quote_map_to_exact_source_substring(self):
        member, raw, root = bt.main_document(archive(), RECEIPT, CORP)
        self.assertEqual(member, RECEIPT + ".xml")
        self.assertEqual(raw, xml_document())
        blocks = bt.extract_blocks(root)
        evidence = bt.extract_evidence(blocks)
        indexed = {b["id"]: b for b in blocks}
        for category in bt.CATEGORIES:
            self.assertGreater(evidence[category]["shown_count"], 0, category)
            for quote in evidence[category]["quotes"]:
                block = indexed[quote["block_id"]]
                self.assertEqual(quote["quote"], block["text"][quote["start"]:quote["end"]])
                self.assertEqual(quote["block_text_sha256"], hashlib.sha256(block["text"].encode()).hexdigest())
        self.assertTrue(evidence["performance"]["quotes"][0]["forward_looking_language"])
        self.assertTrue(evidence["performance"]["quotes"][0]["reason_language_found"])
        self.assertIn(" | ", evidence["products"]["quotes"][0]["quote"])

    def test_failed_main_file_is_never_replaced_with_a_matching_attachment(self):
        for xml in (xml_document(corp="00000001"), xml_document(title="분기보고서"), b"<broken>"):
            with self.subTest(xml=xml[:30]), self.assertRaises(CollectionError):
                bt.main_document(archive(xml, extras={"attachment.xml": xml_document()}), RECEIPT, CORP)

    def test_unsafe_paths_entities_expansion_limits_and_ambiguous_main_are_rejected(self):
        with self.assertRaises(CollectionError):
            bt.main_document(archive(extras={"../escape.xml": "x"}), RECEIPT, CORP)
        with self.assertRaisesRegex(CollectionError, "DTD"):
            bt.validated_root(b'<!DOCTYPE DOCUMENT [<!ENTITY x "abc">]><DOCUMENT>&x;</DOCUMENT>')
        with self.assertRaises(CollectionError):
            bt.validated_root(xml_document().decode().encode("utf-16"))
        with patch("business_text.MAX_EXPANDED", 10), self.assertRaises(CollectionError):
            bt.main_document(archive(), RECEIPT, CORP)
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as z:
            z.writestr("a.xml", xml_document())
            z.writestr("b.xml", xml_document())
        with self.assertRaises(CollectionError):
            bt.main_document(raw.getvalue(), RECEIPT, CORP)

    def test_cap_and_truncation_are_explicit_and_html_is_not_executed(self):
        extra = "".join(f"<P>영업이익은 수요 증가의 영향으로 개선되었습니다. {'추가 설명 ' * 300}&lt;script&gt;bad&lt;/script&gt;</P>" for _ in range(8))
        _, _, root = bt.main_document(archive(xml_document(extra=extra)), RECEIPT, CORP)
        blocks = bt.extract_blocks(root)
        evidence = bt.extract_evidence(blocks)
        self.assertEqual(evidence["performance"]["candidate_count"], 9)
        self.assertEqual(evidence["performance"]["shown_count"], 5)
        self.assertTrue(evidence["performance"]["quotes"][1]["is_partial_block"])
        source = bt.source_html(blocks, "<script>title</script>")
        self.assertNotIn("<script>", source)
        self.assertIn("&lt;script&gt;", source)

    def test_unrelated_content_is_not_business_evidence(self):
        xml = xml_document().replace("II. 사업의 내용".encode(), "I. 회사의 개요".encode())
        xml = xml.replace("1. 사업의 개요".encode(), "1. 배당에 관한 사항".encode())
        _, _, root = bt.main_document(archive(xml), RECEIPT, CORP)
        evidence = bt.extract_evidence(bt.extract_blocks(root))
        self.assertEqual(evidence["overview"]["shown_count"], 0)
        self.assertEqual(evidence["products"]["shown_count"], 0)

    def test_temporal_selection_excludes_same_day_corrections_and_never_fills_latest_failure(self):
        first = {**filing(), "period_end": "2024-12-31", "status": "read"}
        correction = {**filing("20250312000001", "2025-03-12", "[기재정정]사업보고서 (2024.12)"),
                      "period_end": "2024-12-31", "status": "unavailable"}
        self.assertEqual(bt.evidence_as_of([first, correction], date(2025, 3, 12))["receipt_no"], RECEIPT)
        self.assertEqual(bt.evidence_as_of([first, correction], date(2025, 3, 13))["reason"], "latest_report_unreadable")
        duplicate = {**first, "receipt_no": "20250311000002"}
        self.assertEqual(bt.evidence_as_of([first, duplicate], date(2025, 3, 12))["reason"], "ambiguous_same_day_reports")
        self.assertEqual(bt.evidence_as_of([first], date(2025, 3, 11))["reason"], "no_previous_annual_report")
        old_correction = {**first, "receipt_date": "2025-03-14", "period_end": "2023-12-31"}
        self.assertEqual(bt.evidence_as_of([first, old_correction], date(2025, 3, 15))["receipt_no"], RECEIPT)

    def test_document_endpoint_accepts_zip_and_redacts_errors_without_authenticated_urls(self):
        raw = archive()
        client, opener = client_with(raw)
        self.assertEqual(client.original_document(RECEIPT), raw)
        from urllib.parse import urlsplit, parse_qs
        request = opener.open.call_args.args[0]
        self.assertEqual(urlsplit(request.full_url).path, "/api/document.xml")
        self.assertEqual(parse_qs(urlsplit(request.full_url).query)["rcept_no"], [RECEIPT])
        client, _ = client_with(b"<result><status>010</status><message>PRIVATE_KEY</message></result>")
        with self.assertRaises(DartError) as error:
            client.original_document(RECEIPT)
        self.assertNotIn("PRIVATE_KEY", str(error.exception))
        with self.assertRaises(DartError):
            client.original_document("../bad")


class BusinessWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "research" / "sample"
        self.source.mkdir(parents=True)
        batch = {"schema_version": 1, "created_at": "2026-10-28T10:00:00+09:00", "requested_start": "2023-10-26",
                 "requested_end": "2026-10-26", "stocks": [{"code": "005930"}]}
        (self.source / "batch.json").write_text(json.dumps(batch), encoding="utf-8")
        (self.root / "forward").mkdir()
        (self.root / "forward" / "sentinel.txt").write_text("frozen model and journal", encoding="utf-8")
        self.client = Mock()
        self.client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "예시회사"}}
        self.client.company_identifiers.return_value = {CORP}
        self.client.disclosures.return_value = [filing()]
        self.client.original_document.return_value = archive()

    def run_business(self):
        return business.run(self.source, self.root / "business", self.root / "business_documents", self.client, progress=lambda _: None)

    def test_end_to_end_keeps_frozen_journal_and_source_and_reuses_hash_verified_documents(self):
        original = (self.source / "batch.json").read_bytes()
        with patch("learning.fit", side_effect=AssertionError("no training")), patch("observe.run_cycle", side_effect=AssertionError("no forecasts")):
            target, summary = self.run_business()
            next_target, repeated = self.run_business()
        self.assertEqual(summary["status"], "collected")
        self.assertEqual(summary["read_reports"], 1)
        self.assertNotEqual(target, next_target)
        self.client.original_document.assert_called_once_with(RECEIPT)
        self.client.company_identifiers.assert_called_with(CORP, "005930")
        self.assertEqual((self.source / "batch.json").read_bytes(), original)
        self.assertEqual((self.root / "forward" / "sentinel.txt").read_text(), "frozen model and journal")
        self.assertFalse(summary["model_inputs_modified"])
        self.assertFalse(summary["causal_claim_verified"])
        event = summary["stocks"][0]["selected"]["event"]
        blocks_path = target / event["source_blocks"]
        self.assertEqual(hashlib.sha256(blocks_path.read_bytes()).hexdigest(), event["source_blocks_sha256"])
        self.assertIn("추출 본문의 해당 문단 보기", (target / "report.html").read_text(encoding="utf-8"))
        self.assertTrue((target / event["source_html"]).exists())

    def test_corrupted_cache_is_preserved_and_refetched(self):
        self.run_business()
        original = next((self.root / "business_documents").glob("*/original.zip"))
        original.write_bytes(b"corrupted")
        _, summary = self.run_business()
        self.assertEqual(summary["status"], "collected")
        self.assertEqual(original.read_bytes(), b"corrupted")
        self.assertEqual(len(list((self.root / "business_documents").glob("*/original.zip"))), 2)
        self.assertEqual(self.client.original_document.call_count, 2)

    def test_cached_literal_character_document_is_read_without_rewrite_or_redownload(self):
        xml = xml_document(extra='<P>R&D 및 S&P 설명 < TV 시장점유율 추이 > <이사ㆍ감사 전체의 보수현황></P>')
        raw = archive(xml)
        self.client.original_document.return_value = raw
        target, summary = self.run_business()
        event = summary['stocks'][0]['selected']['event']
        parsing = event['xml_parsing']
        self.assertEqual(parsing['literal_ampersands_encoded'], 2)
        self.assertEqual(parsing['literal_less_than_encoded'], 2)
        self.assertEqual(event['xml_sha256'], hashlib.sha256(xml).hexdigest())
        cache_file = next((self.root / 'business_documents').glob('*/original.zip'))
        self.assertEqual(cache_file.read_bytes(), raw)
        self.run_business()
        self.client.original_document.assert_called_once_with(RECEIPT)
        self.assertEqual(cache_file.read_bytes(), raw)
        self.assertIn('R&amp;D', (target / event['source_html']).read_text(encoding='utf-8'))

    def test_new_label_and_eng_formats_flow_through_cached_evidence_report_without_original_changes(self):
        xml = xml_document(extra='<P>지원 제도 <Connected Work> 및 표제 <당기말></P>'
                                 '<TABLE><TR><TE ENG=""Snow Corporation" WIDTH="100">회사명</TE></TR></TABLE>')
        original = archive(xml)
        self.client.original_document.return_value = original
        target, summary = self.run_business()
        self.assertEqual(summary['status'], 'collected')
        event = summary['stocks'][0]['selected']['event']
        self.assertEqual(event['xml_parsing']['eng_attributes_encoded'], 1)
        self.assertEqual(event['xml_parsing']['literal_labels_encoded']['당기말'], 1)
        self.assertEqual(event['xml_sha256'], hashlib.sha256(xml).hexdigest())
        cached = next((self.root / 'business_documents').glob('*/original.zip'))
        self.assertEqual(cached.read_bytes(), original)
        self.run_business()
        self.client.original_document.assert_called_once()
        self.assertEqual(cached.read_bytes(), original)
        self.assertIn('&lt;Connected Work&gt;', (target / event['source_html']).read_text(encoding='utf-8'))

    def test_unreadable_latest_correction_blocks_previous_success_and_keeps_all_targets(self):
        correction = filing("20260311000001", "2026-03-11", "[기재정정]사업보고서 (2024.12)")
        self.client.disclosures.return_value = [filing(), correction]
        self.client.original_document.side_effect = [archive(), archive(xml_document(corp="00000001"), receipt=correction["receipt_no"])]
        target, summary = self.run_business()
        self.assertEqual(summary["read_reports"], 1)
        self.assertEqual(summary["target_reports"], 2)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["stocks"][0]["selected"]["reason"], "latest_report_unreadable")
        self.assertIn("이전 원문으로 대신 채우지 않았습니다", (target / "report.html").read_text(encoding="utf-8"))

    def test_quota_failure_stops_receipt_retries_and_remains_partial(self):
        self.client.disclosures.return_value = [filing(), filing("20260311000001", "2026-03-11", "사업보고서 (2025.12)")]
        self.client.original_document.side_effect = DartError("DART 요청 한도를 초과했습니다.", code="020")
        _, summary = self.run_business()
        self.client.original_document.assert_called_once()
        self.assertEqual(summary["target_reports"], 2)
        self.assertEqual(summary["read_reports"], 0)
        self.assertEqual(summary["stocks"][0]["selected"]["reason"], "latest_report_unreadable")

    def test_private_errors_are_not_saved_and_browser_text_is_escaped(self):
        self.client.companies.return_value["005930"]["corp_name"] = "<script>company</script>"
        self.client.original_document.side_effect = RuntimeError("PRIVATE_API_KEY")
        target, summary = self.run_business()
        page = (target / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_API_KEY", (target / "evidence.json").read_text(encoding="utf-8"))
        self.assertNotIn("<script>company</script>", page)
        self.assertIn("&lt;script&gt;company&lt;/script&gt;", page)

    def test_report_write_failure_preserves_cache_and_no_success_folder(self):
        with patch("business.render_report", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_business()
        self.assertEqual(list((self.root / "business").iterdir()), [])
        self.assertTrue(list((self.root / "business_documents").glob("*/original.zip")))
        self.assertEqual((self.root / "forward" / "sentinel.txt").read_text(), "frozen model and journal")

    def test_cli_reuses_existing_key_and_research_without_prompts(self):
        output = io.StringIO()
        with patch("business.stored_key", return_value="existing_local_key"), patch("business.DartClient", return_value=self.client), redirect_stdout(output):
            status = business.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(status, 0)
        self.assertIn("사업·실적 설명 수집 완료", output.getvalue())


if __name__ == "__main__":
    unittest.main()
