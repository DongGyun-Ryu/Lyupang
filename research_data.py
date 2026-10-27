"""Read verified snapshots; separate past-only calculations from future outcomes."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from lyupang import CSV_COLUMNS, CollectionError, DailyBar, parse_date

HORIZONS = (5, 20, 60)
FEATURE_COLUMNS = ("code", "as_of_date", "available_from", "close", "return_5_pct", "return_20_pct",
                   "return_60_pct", "volume_ratio_previous_20", "quality_issue", "dart_available", "disclosures_previous_7_days")
OUTCOME_COLUMNS = ("code", "as_of_date", "observed_sessions", "status", "entry_date", "entry_open", "exit_date",
                   "exit_close", "price_return_pct", "max_rise_pct", "max_fall_pct", "direction")


def load_snapshot(folder: Path) -> tuple[dict, list[DailyBar]]:
    metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    content = (folder / "daily.csv").read_bytes()
    if metadata.get("schema_version") != 1 or metadata.get("status") != "collected":
        raise CollectionError("재사용할 수 없는 데이터 형식입니다.")
    if hashlib.sha256(content).hexdigest() != metadata.get("csv_sha256"):
        raise CollectionError("수집 파일의 무결성 검사에 실패했습니다.")
    bars = []
    with (folder / "daily.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_COLUMNS.values()):
            raise CollectionError("수집 파일의 열 구성이 다릅니다.")
        for row in reader:
            code = row["종목코드"]
            if code != metadata.get("code"):
                raise CollectionError("수집 파일의 종목코드가 기록과 다릅니다.")
            bar = DailyBar.from_wire(code, {
                "dt": parse_date(row["거래일"]).strftime("%Y%m%d"), "open_pric": row["시가_원"],
                "high_pric": row["고가_원"], "low_pric": row["저가_원"], "cur_prc": row["종가_원"],
                "trde_qty": row["거래량_주"], "trde_prica": row["거래대금_백만원"],
            })
            if row["거래량_상한값의심"] != str(bar.volume_capped) or row["거래량_0"] != str(bar.zero_volume):
                raise CollectionError("수집 파일의 품질 표시가 값과 일치하지 않습니다.")
            bars.append(bar)
    days = [bar.date for bar in bars]
    if not bars or days != sorted(set(days)) or metadata.get("row_count") != len(bars):
        raise CollectionError("수집 파일의 날짜·행 수가 기록과 일치하지 않습니다.")
    if days[0] != metadata.get("observed_start") or days[-1] != metadata.get("observed_end"):
        raise CollectionError("수집 파일의 실제 기간이 기록과 일치하지 않습니다.")
    return metadata, bars


def find_cached(root: Path, code: str, start: date, end: date) -> tuple[Path, list[DailyBar]] | None:
    candidates = []
    if not root.exists():
        return None
    for folder in root.iterdir():
        if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
            continue
        try:
            metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if (metadata.get("code") != code or metadata.get("requested_start") != start.isoformat()
                    or metadata.get("requested_end") != end.isoformat() or metadata.get("adjusted") is not True
                    or metadata.get("adjustment_base_date") != end.isoformat()):
                continue
            captured = datetime.fromisoformat(metadata["retrieved_at"])
            if captured.tzinfo is None:
                continue
            candidates.append((captured, folder))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for _, folder in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            _, bars = load_snapshot(folder)
            return folder, bars
        except (OSError, ValueError, KeyError, TypeError, CollectionError):
            continue
    return None


def _pct(value: int, basis: int) -> float | None:
    return round((value / basis - 1) * 100, 6) if basis > 0 else None


def build_cases(bars: list[DailyBar], disclosures: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """60 observed sessions of history; outcomes enter a different file, never the features."""
    features, outcomes = [], []
    if len({bar.code for bar in bars}) > 1 or [b.date for b in bars] != sorted({b.date for b in bars}):
        raise CollectionError("사례 생성에는 한 종목의 정렬된 일별 자료가 필요합니다.")
    filing_dates = [parse_date(record["receipt_date"]) for record in disclosures] if disclosures is not None else []
    for index in range(60, len(bars)):
        bar = bars[index]
        as_of = parse_date(bar.date)
        previous = bars[index - 20:index]
        volumes_ok = all(not b.volume_capped for b in previous) and not bar.volume_capped
        average = sum(b.volume for b in previous) / 20
        feature = {
            "code": bar.code, "as_of_date": bar.date,
            "available_from": (as_of + timedelta(days=1)).isoformat() + "T00:00:00+09:00",
            "close": bar.close,
            "volume_ratio_previous_20": round(bar.volume / average, 6) if volumes_ok and average else None,
            "quality_issue": bar.volume_capped or bar.zero_volume or bar.close == 0,
            "dart_available": disclosures is not None,
            # Receipt time is unavailable: conservatively exclude same-day disclosures.
            "disclosures_previous_7_days": sum(as_of - timedelta(days=7) <= d < as_of for d in filing_dates)
            if disclosures is not None else None,
        }
        for horizon in HORIZONS:
            window = bars[index - horizon:index + 1]
            feature[f"return_{horizon}_pct"] = _pct(bar.close, window[0].close) if all(b.close > 0 for b in window) else None
        features.append(feature)
        for horizon in HORIZONS:
            outcome = dict.fromkeys(OUTCOME_COLUMNS)
            outcome.update(code=bar.code, as_of_date=bar.date, observed_sessions=horizon, status="pending")
            if index + horizon < len(bars):
                future = bars[index + 1:index + horizon + 1]
                if any(b.open <= 0 or b.close <= 0 or b.zero_volume or b.volume_capped for b in future):
                    outcome["status"] = "quality_issue"
                else:
                    first, last = future[0], future[-1]
                    change = _pct(last.close, first.open)
                    outcome.update(status="observed", entry_date=first.date, entry_open=first.open,
                                   exit_date=last.date, exit_close=last.close, price_return_pct=change,
                                   max_rise_pct=_pct(max(b.high for b in future), first.open),
                                   max_fall_pct=_pct(min(b.low for b in future), first.open),
                                   direction="up" if last.close > first.open else "down" if last.close < first.open else "flat")
            outcomes.append(outcome)
    return features, outcomes


def write_table(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
