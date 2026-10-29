import copy
import hashlib
import unittest

from financial_crosscheck import compare_amount
from lyupang import CollectionError
from report_claims import extract_claims
from report_context import ContextIndex
from test_financial_crosscheck import financial_doc


HEAD = ["IV. 이사의 경영진단 및 분석의견", "3. 재무상태 및 영업실적"]


def block(text, i=1, *, headings=None, kind="paragraph"):
    return {"id": f"b{i:06d}", "kind": kind, "text": text, "headings": headings if headings is not None else HEAD,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def case(blocks, needle, *, name="삼성전자", metric="revenue", occurrence=0):
    event = {"receipt_no": "20240312000001", "receipt_date": "2024-03-12", "period_end": "2023-12-31", "available_from": "2024-03-13T00:00:00+09:00"}
    claims, _ = extract_claims(blocks, event, "005930", name)
    c = next(c for c in claims if needle in c["text"])
    q = [q for q in c["measurements"] if q["metric"] == metric and q["unit"] == "KRW"][occurrence]
    return c, q, ContextIndex(blocks, name)


class ReportContextTests(unittest.TestCase):
    def test_uniform_same_paragraph_links_year_basis_with_exact_sources(self):
        blocks = [block("2023년 당사의 연결 기준 매출은 200원입니다. 당사의 순이익은 20원입니다.")]
        c, q, idx = case(blocks, "당사의 순이익", metric="net_profit")
        before = copy.deepcopy(c)
        resolved = idx.resolve(c, q)
        self.assertEqual((resolved["year"], resolved["scope"]), (2023, "CFS"))
        self.assertEqual(compare_amount(c, q, financial_doc(), context=resolved)["status"], "exact_match")
        self.assertEqual(c, before)
        for e in resolved["evidence"]:
            b = idx.by_id[e["block_id"]]
            self.assertEqual(b["text"][e["start"]:e["end"]], e["text"])
            self.assertEqual(e["block_sha256"], b["text_sha256"])

    def test_explicit_mixed_clauses_are_separated_per_quantity(self):
        blocks = [block("2023년 당사는 연결기준 매출 200원, 별도기준 매출 100원을 기록하였습니다.")]
        c, q, idx = case(blocks, "2023년 당사는")
        first, second = [q for q in c["measurements"] if q["unit"] == "KRW"]
        a, b = idx.resolve(c, first), idx.resolve(c, second)
        self.assertEqual((a["scope"], b["scope"]), ("CFS", "OFS"))
        self.assertEqual(a["basis_method"], "clause_explicit")
        self.assertEqual(compare_amount(c, first, financial_doc())["status"], "not_comparable")
        self.assertEqual(compare_amount(c, first, financial_doc(), context=a)["status"], "exact_match")
        self.assertEqual(compare_amount(c, second, financial_doc(), context=b)["status"], "not_comparable")

    def test_duplicate_same_metric_within_one_basis_does_not_become_approved(self):
        blocks = [block("2023년 당사는 연결기준 매출 200원, 매출 300원, 별도기준 매출 100원을 기록하였습니다.")]
        c, q, idx = case(blocks, "2023년 당사는")
        context = idx.resolve(c, q)
        self.assertEqual(context["same_metric_scope_amounts"], 2)
        self.assertIn("statement_requires_review", compare_amount(c, q, financial_doc(), context=context)["reasons"])

    def test_active_heading_supplies_year_and_scope_but_title_year_is_not_default(self):
        heading = "2023년 연결 기준 영업실적"
        blocks = [block(heading, kind="heading", headings=[heading]), block("당사의 매출은 200원입니다.", 2, headings=[heading])]
        # Add a supported business/management heading to the source path.
        for b in blocks: b["headings"] = ["IV. 이사의 경영진단 및 분석의견", heading]
        c, q, idx = case(blocks, "당사의 매출")
        self.assertEqual(idx.resolve(c, q)["year"], 2023)
        self.assertEqual(idx.resolve(c, q)["scope"], "CFS")
        self.assertEqual(compare_amount(c, q, financial_doc(), context=idx.resolve(c, q))["status"], "exact_match")
        blocks[0] = block("2023년 사업보고서", kind="heading", headings=blocks[0]["headings"])
        self.assertIsNone(ContextIndex(blocks, "삼성전자").resolve(c, q)["year"])

    def test_adjacent_declaration_requires_direction_and_same_heading(self):
        for declaration, expected in (("아래 실적은 2023년 연결기준으로 작성하였습니다.", "CFS"),
                                      ("다른 실적은 2023년 연결기준입니다.", None),
                                      ("아래 표는 2023년 연결기준입니다.", None)):
            blocks = [block(declaration), block("2023년 당사의 매출은 200원입니다.", 2)]
            c, q, idx = case(blocks, "당사의 매출")
            self.assertEqual(idx.resolve(c, q)["scope"], expected)
        blocks[0] = block("아래 실적은 2023년 연결기준입니다.", headings=["다른 절"])
        self.assertIsNone(ContextIndex(blocks, "삼성전자").resolve(c, q)["scope"])

    def test_following_declaration_can_refer_to_above_not_below(self):
        blocks = [block("2023년 당사의 매출은 200원입니다."), block("상기 실적은 연결기준으로 작성하였습니다.", 2)]
        c, q, idx = case(blocks, "당사의 매출")
        self.assertEqual(idx.resolve(c, q)["scope"], "CFS")
        blocks[1] = block("아래 실적은 연결기준입니다.", 2)
        self.assertIsNone(ContextIndex(blocks, "삼성전자").resolve(c, q)["scope"])

    def test_conflicting_adjacent_declarations_are_not_chosen_by_financial_match(self):
        blocks = [block("아래 실적은 연결기준입니다."), block("2023년 당사의 매출은 200원입니다.", 2), block("상기 실적은 별도기준입니다.", 3)]
        c, q, idx = case(blocks, "당사의 매출")
        self.assertIsNone(idx.resolve(c, q)["scope"])

    def test_other_company_forecast_reference_and_other_year_do_not_transfer(self):
        for text in ("Alphabet은 2023년 연결기준 매출을 기록했습니다.", "2023년 연결기준 매출을 예상합니다.",
                     "2023년 연결기준 매출은 다른 자료를 참고했습니다.", "2022년 연결기준 매출을 기록했습니다."):
            blocks = [block(text+" 2023년 당사의 매출은 200원입니다.")]
            c, q, idx = case(blocks, "2023년 당사의 매출")
            self.assertIsNone(idx.resolve(c, q)["scope"])

    def test_chapter_declaration_uses_its_own_sentence_not_nearby_separate_discussion(self):
        blocks = [block("별도재무제표도 작성합니다. 본 '이사의 경영진단 및 분석 의견'은 연결재무제표의 데이터를 기준으로 작성되었습니다.", headings=[HEAD[0], "1. 주의사항"]),
                  block(HEAD[1], 2, kind="heading"), block("2023년 당사의 매출은 200원입니다.", 3)]
        c, q, idx = case(blocks, "당사의 매출")
        result = idx.resolve(c, q)
        self.assertEqual(result["scope"], "CFS")
        self.assertIn("explicit_chapter_declaration", [e["method"] for e in result["evidence"]])
        blocks[0] = block(blocks[0]["text"], headings=["다른 장의 경영진단"])
        self.assertIsNone(ContextIndex(blocks, "삼성전자").resolve(c, q)["scope"])

    def test_explicit_fiscal_year_pair_is_required_and_conflicts_hold(self):
        blocks = [block("2023년(제54기) 당사의 매출은 200원입니다."), block("제54기 당사의 연결 기준 매출은 200원입니다.", 2)]
        c, q, idx = case(blocks, "제54기 당사의")
        result = idx.resolve(c, q)
        self.assertEqual(result["year"], 2023)
        self.assertEqual(result["evidence"][0]["method"], "explicit_fiscal_year_pair")
        self.assertEqual(compare_amount(c, q, financial_doc(), context=result)["status"], "exact_match")
        blocks.insert(1, block("2022년(제54기) 당사의 매출은 100원입니다.", 3))
        self.assertIsNone(ContextIndex(blocks, "삼성전자").resolve(c, q)["year"])
        self.assertIsNone(ContextIndex(blocks[-1:], "삼성전자").resolve(c, q)["year"])

    def test_direct_basis_not_replaced_by_a_conflicting_context(self):
        blocks = [block("아래 실적은 연결기준입니다."), block("2023년 당사의 별도기준 매출은 200원입니다.", 2)]
        c, q, idx = case(blocks, "당사의 별도기준")
        self.assertEqual(idx.resolve(c, q)["scope"], "OFS")

    def test_quarter_and_mixed_year_never_become_annual(self):
        for text in ("2023년 1분기 당사의 연결기준 매출은 200원입니다.", "2022년과 2023년 당사의 연결기준 매출은 200원입니다."):
            blocks = [block(text)]; c, q, idx = case(blocks, "당사의")
            self.assertIsNone(idx.resolve(c, q)["year"])

    def test_table_or_unknown_company_context_does_not_supply_basis(self):
        blocks = [block("2023년 연결기준 매출 200원", kind="table"), block("2023년 당사의 매출은 200원입니다.", 2)]
        c, q, idx = case(blocks, "당사의 매출")
        self.assertIsNone(idx.resolve(c, q)["scope"])

    def test_mixed_paragraph_blocks_chapter_default_for_an_unmarked_sentence(self):
        blocks = [block("본 이사의 경영진단 및 분석 의견은 연결재무제표를 기준으로 작성되었습니다.", headings=[HEAD[0]]),
                  block("연결기준 매출을 기록했습니다. 별도기준 순이익을 기록했습니다. 2023년 당사의 매출은 200원입니다.", 2)]
        c, q, idx = case(blocks, "2023년 당사의 매출")
        self.assertIsNone(idx.resolve(c, q)["scope"])

    def test_context_does_not_depend_on_normalized_amount(self):
        blocks = [block("아래 실적은 연결기준입니다."), block("2023년 당사의 매출은 200원입니다.", 2)]
        c, q, idx = case(blocks, "당사의 매출")
        expected = idx.resolve(c, q)
        q["normalized_value"] = "999999999999999"
        self.assertEqual(idx.resolve(c, q), expected)

    def test_mixed_negated_basis_is_not_treated_as_a_plain_enumeration(self):
        blocks = [block("2023년 당사의 연결기준이 아닌 별도기준 매출은 200원입니다.")]
        c, q, idx = case(blocks, "당사의 연결기준")
        context = idx.resolve(c, q)
        self.assertIsNone(context["scope"])
        self.assertEqual(compare_amount(c, q, financial_doc(), context=context)["status"], "not_comparable")

    def test_wrong_document_and_changed_context_hash_fail_closed(self):
        blocks = [block("아래 실적은 연결기준입니다."), block("2023년 당사의 매출은 200원입니다.", 2)]
        c, q, idx = case(blocks, "당사의 매출")
        wrong = ContextIndex(blocks, "삼성전자", code="005930", receipt="20240313000002")
        with self.assertRaises(CollectionError): wrong.resolve(c, q)
        blocks[0]["text"] = "아래 실적은 별도기준입니다."
        with self.assertRaises(CollectionError): ContextIndex(blocks, "삼성전자")


if __name__ == "__main__": unittest.main()
