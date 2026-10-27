"""Historical KRX daily bars. This module contains no order API."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")
TABLE = "stk_dt_pole_chart_qry"
API_ID = "ka10081"
API_PATH = "/api/dostk/chart"
VOLUME_CAP = 2**32 - 1


class CollectionError(Exception):
    """A user-facing, credential-free error."""


def parse_date(value: str) -> date:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise CollectionError("날짜는 2026-10-26처럼 연-월-일로 입력해주세요.")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise CollectionError("존재하지 않는 날짜입니다.") from None


def default_period(today: date | None = None) -> tuple[date, date]:
    end = (today or datetime.now(SEOUL).date()) - timedelta(days=1)
    try:
        start = end.replace(year=end.year - 3)
    except ValueError:
        start = end.replace(year=end.year - 3, day=28)
    return start, end


def validate_request(code: str, start: date, end: date, *, today: date | None = None) -> None:
    if not re.fullmatch(r"[0-9]{6}", code):
        raise CollectionError("종목코드는 005930처럼 숫자 6자리로 입력해주세요.")
    if start > end:
        raise CollectionError("시작일은 종료일보다 늦을 수 없습니다.")
    if end >= (today or datetime.now(SEOUL).date()):
        raise CollectionError("오늘의 미완성 일봉을 제외하도록 종료일을 한국 시간 기준 어제까지로 정해주세요.")


def _integer(value: object, field: str, *, price: bool = False) -> int:
    # Kiwoom uses signs on prices to express price direction, not negative KRW.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CollectionError(f"가격·거래량 응답의 {field} 형식이 올바르지 않습니다.")
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise CollectionError(f"가격·거래량 응답의 {field} 값이 비어 있거나 올바르지 않습니다.")
    number = int(text)
    if price:
        return abs(number)
    if number < 0:
        raise CollectionError(f"가격·거래량 응답의 {field} 값이 음수입니다.")
    return number


@dataclass(frozen=True)
class DailyBar:
    code: str
    date: str
    open: int
    high: int
    low: int
    close: int
    volume: int
    turnover_million_krw: int | None
    volume_capped: bool
    zero_volume: bool

    @classmethod
    def from_wire(cls, code: str, row: dict) -> DailyBar:
        day = row.get("dt")
        if not isinstance(day, str) or not re.fullmatch(r"[0-9]{8}", day):
            raise CollectionError("키움 응답에 잘못된 거래일이 있습니다.")
        try:
            parsed_day = datetime.strptime(day, "%Y%m%d").date()
        except ValueError:
            raise CollectionError("키움 응답에 존재하지 않는 거래일이 있습니다.") from None
        values = {key: _integer(row.get(field), field, price=True) for key, field in (
            ("open", "open_pric"), ("high", "high_pric"), ("low", "low_pric"), ("close", "cur_prc"),
        )}
        volume = _integer(row.get("trde_qty"), "trde_qty")
        if not (values["low"] <= min(values["open"], values["close"])
                <= max(values["open"], values["close"]) <= values["high"]):
            raise CollectionError("키움 응답의 시가·고가·저가·종가 관계가 맞지 않습니다.")
        if any(v == 0 for v in values.values()) and not (all(v == 0 for v in values.values()) and volume == 0):
            raise CollectionError("키움 응답에 일관되지 않은 0원 가격이 있습니다.")
        turnover = row.get("trde_prica")
        return cls(code, parsed_day.isoformat(), **values, volume=volume,
                   turnover_million_krw=None if turnover in (None, "") else _integer(turnover, "trde_prica"),
                   volume_capped=volume == VOLUME_CAP, zero_volume=volume == 0)


@dataclass(frozen=True)
class Page:
    rows: list[dict]
    has_next: bool
    next_key: str | None


class DailySource(Protocol):
    def fetch(self, code: str, end: date, *, adjusted: bool, next_key: str | None) -> Page: ...


class KiwoomSource:
    """Only the fixed ka10081 endpoint is used; saved Windows credentials stay in the SDK."""
    def __init__(self, profile: str = "kiwoom-real", *, client=None):
        if client is None:
            try:
                from kiwoom import get_client
            except ImportError:
                raise CollectionError("키움 도구가 있는 파이썬으로 실행해주세요. Windows에서는 START.cmd를 사용하세요.") from None
            client = get_client(profile=profile, timeout_seconds=30)
        self._client = client

    def fetch(self, code: str, end: date, *, adjusted: bool, next_key: str | None) -> Page:
        response = self._client.fetch_page(
            api_id=API_ID, path=API_PATH,
            body={"stk_cd": code, "base_dt": end.strftime("%Y%m%d"), "upd_stkpc_tp": "1" if adjusted else "0"},
            cont_yn="Y" if next_key is not None else None, next_key=next_key,
        )
        body = response.body
        if not isinstance(body, dict) or body.get("return_code") not in (0, "0"):
            raise CollectionError("키움에서 일봉 조회를 처리하지 못했습니다.")
        if body.get("stk_cd") != code:
            raise CollectionError("요청한 종목과 응답 종목이 다릅니다.")
        rows = body.get(TABLE)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise CollectionError("키움 응답에서 일봉 목록을 확인하지 못했습니다.")
        return Page(rows, response.continuation.has_next, response.continuation.next_key)


@dataclass(frozen=True)
class Collection:
    bars: list[DailyBar]
    pages: int
    stop_reason: str
    code: str
    start: date
    end: date
    adjusted: bool


def collect(source: DailySource, code: str, start: date, end: date, *, adjusted: bool = True,
            max_pages: int = 100, delay: float = 0.6, sleep: Callable = time.sleep,
            progress: Callable[[int, int], None] | None = None) -> Collection:
    validate_request(code, start, end)
    if max_pages < 1 or delay < 0:
        raise CollectionError("조회 횟수와 대기 시간 설정이 올바르지 않습니다.")
    seen: dict[str, DailyBar] = {}
    cursors: set[str] = set()
    next_key = None
    previous_oldest = None
    for page_number in range(1, max_pages + 1):
        page = source.fetch(code, end, adjusted=adjusted, next_key=next_key)
        parsed = [DailyBar.from_wire(code, row) for row in page.rows]
        days = [bar.date for bar in parsed]
        if days != sorted(days, reverse=True):
            raise CollectionError("일봉 응답의 날짜 순서가 맞지 않아 수집을 중단했습니다.")
        if any(day > end.isoformat() for day in days):
            raise CollectionError("조회 종료일 이후의 데이터가 응답에 포함되어 있습니다.")
        new = 0
        for bar in parsed:
            old = seen.get(bar.date)
            if old is not None:
                if old != bar:
                    raise CollectionError("같은 거래일에 서로 다른 값이 응답되어 수집을 중단했습니다.")
            else:
                if previous_oldest is not None and bar.date > previous_oldest:
                    raise CollectionError("연속조회 날짜가 이전으로 진행되지 않았습니다.")
                seen[bar.date] = bar
                new += 1
        if days:
            previous_oldest = days[-1]
        if progress:
            progress(page_number, sum(start.isoformat() <= d <= end.isoformat() for d in seen))
        if days and days[-1] <= start.isoformat():
            stop_reason = "requested_start_reached"
            break
        if not page.has_next:
            stop_reason = "source_history_exhausted"
            break
        if not new:
            raise CollectionError("연속조회에 새 거래일이 없어 중단했습니다. 수집 완료로 저장하지 않습니다.")
        if not isinstance(page.next_key, str) or not page.next_key.strip() or page.next_key in cursors:
            raise CollectionError("연속조회 키가 없거나 반복되어 중단했습니다. 수집 완료로 저장하지 않습니다.")
        cursors.add(page.next_key)
        next_key = page.next_key
        if page_number == max_pages:
            raise CollectionError("최대 조회 횟수에 도달했습니다. 기간을 줄여 다시 실행해주세요.")
        sleep(delay)
    bars = [bar for day, bar in sorted(seen.items()) if start.isoformat() <= day <= end.isoformat()]
    if not bars:
        raise CollectionError("요청한 기간의 거래 데이터가 없습니다. 종목코드와 기간을 확인해주세요.")
    return Collection(bars, page_number, stop_reason, code, start, end, adjusted)


CSV_COLUMNS = {
    "code": "종목코드", "date": "거래일", "open": "시가_원", "high": "고가_원", "low": "저가_원",
    "close": "종가_원", "volume": "거래량_주", "turnover_million_krw": "거래대금_백만원",
    "volume_capped": "거래량_상한값의심", "zero_volume": "거래량_0",
}


def save_collection(result: Collection, directory: Path, *, now: datetime | None = None) -> Path:
    """Publish a complete snapshot; errors never leave a success-labelled partial CSV."""
    now = now or datetime.now(SEOUL)
    if now.tzinfo is None:
        raise CollectionError("수집 시각에는 시간대가 필요합니다.")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{now.astimezone(SEOUL):%Y%m%d_%H%M%S}_{result.code}_{uuid.uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    try:
        csv_path = staging / "daily.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS.values()))
            writer.writeheader()
            for bar in result.bars:
                writer.writerow({CSV_COLUMNS[k]: v for k, v in asdict(bar).items()})
        quality = {
            "calendar_verified": False,
            "missing_trading_days": "not_verified_against_exchange_calendar",
            "volume_capped_dates": [b.date for b in result.bars if b.volume_capped],
            "zero_volume_dates": [b.date for b in result.bars if b.zero_volume],
        }
        manifest = {
            "schema_version": 1, "status": "collected", "provider": "kiwoom", "api_id": API_ID,
            "market": "KRX", "code": result.code,
            "requested_start": result.start.isoformat(), "requested_end": result.end.isoformat(),
            "observed_start": result.bars[0].date, "observed_end": result.bars[-1].date,
            "row_count": len(result.bars), "pages": result.pages, "stop_reason": result.stop_reason,
            "adjusted": result.adjusted, "adjustment_base_date": result.end.isoformat(),
            "retrieved_at": now.isoformat(), "quality": quality,
            "point_in_time_training_ready": False,
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "notes": [
                "수정주가는 기준일까지의 권리 변동을 반영할 수 있습니다. 과거 당시의 정보로 간주하지 마세요.",
                "거래일 달력 대조는 수행하지 않았습니다. 휴장·거래정지·상장 시점과 누락 여부를 별도 확인해야 합니다.",
                "과거 주가만 수집합니다. 재무·공시·뉴스의 당시 공개 시점과 상장폐지 종목을 포함한 검증은 별도 단계입니다.",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target
