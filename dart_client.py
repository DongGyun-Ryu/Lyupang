"""OpenDART read-only data and local key storage. Never log authenticated URLs."""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta

from lyupang import CollectionError

SERVICE = "Lyupang/OpenDART"
KEY_NAME = "api_key"
BASE = "https://opendart.fss.or.kr/api/"
PATHS = {"corpCode.xml", "list.json", "fnlttSinglAcntAll.json", "company.json", "fnlttXbrl.xml"}
LIMIT = 20 * 1024 * 1024


class DartError(CollectionError):
    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code


def is_annual_report(name) -> bool:
    """Classify returned titles too; a query filter alone is not a response guarantee."""
    return isinstance(name, str) and re.match(r"^(?:\[[^\]]+\]\s*)*사업보고서(?:[\s(（]|$)", name.strip()) is not None


def stored_key() -> str | None:
    if os.environ.get("DART_API_KEY"):
        return os.environ["DART_API_KEY"]
    if os.name != "nt":
        return None
    import keyring
    return keyring.get_password(SERVICE, KEY_NAME)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DartError("DART 인증키 보호를 위해 주소 변경 응답을 따르지 않았습니다.")


class DartClient:
    def __init__(self, key: str, *, opener=None, delay: float = 0.3, sleep=time.sleep, clock=time.monotonic):
        # Cloud proxy bindings can be placeholders; let the supported HTTPS route validate them.
        if not isinstance(key, str) or not key.strip() or len(key) > 4096:
            raise DartError("DART 인증키 설정이 필요합니다.")
        self._key = key
        self._opener = opener or urllib.request.build_opener(NoRedirect())
        self._delay, self._sleep, self._clock = delay, sleep, clock
        self._last_request = None

    def _download(self, path: str, params: dict) -> bytes:
        if path not in PATHS or "crtfc_key" in params:
            raise DartError("허용되지 않은 DART 요청입니다.")
        if self._last_request is not None:
            remaining = self._delay - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()
        query = urllib.parse.urlencode({**params, "crtfc_key": self._key})
        request = urllib.request.Request(BASE + path + "?" + query, headers={"User-Agent": "Lyupang/0.3.1"})
        try:
            with self._opener.open(request, timeout=30) as response:
                payload = response.read(LIMIT + 1)
        except urllib.error.HTTPError as exc:
            raise DartError(f"DART 서버가 요청을 거절했습니다. HTTP {exc.code}.", code=f"HTTP_{exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise DartError("DART 서버에 접속하지 못했습니다.", code="network") from None
        if len(payload) > LIMIT:
            raise DartError("DART 응답이 처리 가능한 크기를 초과했습니다.")
        return payload

    @staticmethod
    def _status(value) -> None:
        if value in ("000", "013"):
            return
        messages = {"010": "등록되지 않은 DART 인증키입니다.", "011": "사용할 수 없는 DART 인증키입니다.",
                    "012": "DART에 등록된 접속 IP를 확인해주세요.", "020": "DART 요청 한도를 초과했습니다.",
                    "800": "DART 서버가 점검 중입니다.", "901": "DART 계정의 사용 가능 상태를 확인해주세요."}
        raise DartError(messages.get(value, "DART가 요청을 처리하지 못했습니다.") if isinstance(value, str)
                        else "DART 응답의 상태 코드가 올바르지 않습니다.", code=value if isinstance(value, str) else None)

    def _json(self, path: str, params: dict) -> dict:
        try:
            data = json.loads(self._download(path, params))
        except (ValueError, UnicodeError):
            raise DartError("DART 응답이 올바른 JSON 형식이 아닙니다.") from None
        if not isinstance(data, dict):
            raise DartError("DART 응답 형식이 올바르지 않습니다.")
        self._status(data.get("status"))
        return data

    def validate_key(self) -> None:
        data = self._json("company.json", {"corp_code": "00126380"})
        if data.get("status") != "000" or data.get("corp_code") != "00126380":
            raise DartError("DART 기업개황 조회로 인증키를 확인하지 못했습니다.")

    def companies(self, codes: list[str]) -> dict[str, dict]:
        payload = self._download("corpCode.xml", {})
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
                names = [n for n in bundle.namelist() if n.upper() == "CORPCODE.XML"]
                if len(names) != 1 or bundle.getinfo(names[0]).file_size > 64 * 1024 * 1024:
                    raise DartError("DART 기업번호 파일 구성이 올바르지 않습니다.")
                xml = bundle.read(names[0])
        except zipfile.BadZipFile:
            try:
                root = ET.fromstring(payload)
                self._status(root.findtext("status"))
            except ET.ParseError:
                pass
            raise DartError("DART 기업번호 압축 파일을 받지 못했습니다.") from None
        if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
            raise DartError("DART 기업번호 XML에 허용되지 않은 선언이 있습니다.")
        found = {}
        try:
            for _, element in ET.iterparse(io.BytesIO(xml), events=("end",)):
                if element.tag != "list":
                    continue
                code = (element.findtext("stock_code") or "").strip()
                if code in codes:
                    corp = (element.findtext("corp_code") or "").strip()
                    if not re.fullmatch(r"[0-9]{8}", corp):
                        raise DartError("DART 기업 고유번호가 올바르지 않습니다.")
                    if code in found and found[code]["corp_code"] != corp:
                        raise DartError("한 종목에 여러 DART 기업번호가 있어 자동 연결하지 않았습니다.")
                    found[code] = {"corp_code": corp, "corp_name": element.findtext("corp_name") or code}
                element.clear()
        except ET.ParseError:
            raise DartError("DART 기업번호 XML 형식이 올바르지 않습니다.") from None
        return found

    def disclosures(self, corp: str, start: date, end: date, *, max_pages: int = 100, annual_only=False) -> list[dict]:
        records, seen = [], set()
        expected_count, expected_pages = None, None
        for page in range(1, max_pages + 1):
            params = {"corp_code": corp, "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"), "last_reprt_at": "N", "sort": "date", "sort_mth": "asc",
                "page_count": 100, "page_no": page}
            if annual_only:
                params["pblntf_detail_ty"] = "A001"
            data = self._json("list.json", params)
            if data["status"] == "013":
                if page != 1:
                    raise DartError("DART 연속조회 중 목록이 사라졌습니다.")
                return []
            try:
                total, pages = int(data["total_count"]), int(data["total_page"])
                actual_page = int(data["page_no"])
            except (KeyError, ValueError, TypeError):
                raise DartError("DART 목록의 페이지 정보가 올바르지 않습니다.") from None
            if page == 1:
                expected_count, expected_pages = total, pages
            if (total, pages) != (expected_count, expected_pages) or actual_page != page or pages < 1 or pages > max_pages:
                raise DartError("DART 목록 크기나 페이지가 달라져 중단했습니다.")
            rows = data.get("list")
            if not isinstance(rows, list) or not rows:
                raise DartError("DART 공시 페이지가 비어 있습니다.")
            for row in rows:
                if not isinstance(row, dict) or row.get("corp_code") != corp:
                    raise DartError("DART 공시의 기업번호가 요청과 다릅니다.")
                receipt = row.get("rcept_no", "")
                if not isinstance(receipt, str) or not re.fullmatch(r"[0-9]{14}", receipt) or receipt in seen:
                    raise DartError("DART 공시 접수번호가 잘못됐거나 반복됐습니다.")
                try:
                    received = datetime.strptime(row["rcept_dt"], "%Y%m%d").date()
                except (KeyError, ValueError, TypeError):
                    raise DartError("DART 접수일자가 올바르지 않습니다.") from None
                if not start <= received <= end:
                    raise DartError("DART 공시 접수일이 요청 기간 밖에 있습니다.")
                seen.add(receipt)
                records.append({"receipt_no": receipt, "receipt_date": received.isoformat(),
                    "available_from": (received + timedelta(days=1)).isoformat() + "T00:00:00+09:00",
                    "report_name": str(row.get("report_nm", ""))[:1000],
                    "url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + receipt})
            if page == pages:
                if len(records) != total:
                    raise DartError("DART 공시 건수가 페이지 기록과 일치하지 않습니다.")
                ordered = sorted(records, key=lambda r: (r["receipt_date"], r["receipt_no"]))
                return [record for record in ordered if is_annual_report(record["report_name"])] if annual_only else ordered
        raise DartError("DART 공시 최대 조회 횟수에 도달했습니다.")

    def company_identifiers(self, corp: str, code: str) -> set[str]:
        data = self._json("company.json", {"corp_code": corp})
        if data.get("status") != "000" or data.get("corp_code") != corp or data.get("stock_code") != code:
            raise DartError("원문 검증용 기업 식별번호를 확인하지 못했습니다.")
        values = {re.sub(r"[^0-9]", "", str(data[key])) for key in ("corp_code", "stock_code", "bizr_no", "jurir_no")
                  if data.get(key) and re.fullmatch(r"[0-9-]+", str(data[key]))}
        return {value for value in values if value}

    def original_xbrl(self, receipt: str) -> bytes:
        if not re.fullmatch(r"[0-9]{14}", receipt):
            raise DartError("재무제표 접수번호가 올바르지 않습니다.")
        payload = self._download("fnlttXbrl.xml", {"rcept_no": receipt, "reprt_code": "11011"})
        if not zipfile.is_zipfile(io.BytesIO(payload)):
            try:
                root = ET.fromstring(payload)
                self._status(root.findtext("status"))
            except ET.ParseError:
                pass
            raise DartError("해당 접수번호의 XBRL 원본 압축 파일이 제공되지 않았습니다.")
        return payload

    def annual_accounts(self, corp: str, year: int) -> dict:
        fields = ("rcept_no", "reprt_code", "bsns_year", "sj_div", "account_id", "account_nm", "thstrm_nm",
                  "thstrm_amount", "frmtrm_nm", "frmtrm_amount", "bfefrmtrm_nm", "bfefrmtrm_amount", "currency")
        for kind in ("CFS", "OFS"):
            data = self._json("fnlttSinglAcntAll.json", {"corp_code": corp, "bsns_year": year, "reprt_code": "11011", "fs_div": kind})
            if data["status"] == "013":
                continue
            rows = data.get("list")
            if not isinstance(rows, list) or not rows:
                raise DartError("DART 재무제표 목록이 올바르지 않습니다.")
            clean = []
            for row in rows:
                if (not isinstance(row, dict) or row.get("corp_code") != corp
                        or str(row.get("bsns_year")) != str(year) or row.get("reprt_code") != "11011"
                        or not re.fullmatch(r"[0-9]{14}", str(row.get("rcept_no", "")))):
                    raise DartError("DART 재무제표의 기업·연도·접수번호가 맞지 않습니다.")
                clean.append({key: str(row.get(key, ""))[:1000] for key in fields})
            return {"status": "collected", "business_year": year, "fs_div": kind, "accounts": clean,
                    "point_in_time_verified": False, "use_in_historical_features": False}
        return {"status": "no_data", "business_year": year, "accounts": [], "point_in_time_verified": False,
                "use_in_historical_features": False}
