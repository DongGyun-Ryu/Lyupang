"""Console entry point. No credentials are accepted on the command line."""
import argparse
import os
import sys
from pathlib import Path

from lyupang import CollectionError, KiwoomSource, collect, default_period, parse_date, save_collection, validate_request


def main(argv=None) -> int:
    start, end = default_period()
    parser = argparse.ArgumentParser(description="Lyupang — 국내 주식 일별 가격·거래량 수집")
    parser.add_argument("--code", help="KRX 종목코드 6자리 (기본: 005930 삼성전자)")
    parser.add_argument("--start", default=start.isoformat(), help="시작일 YYYY-MM-DD (기본: 약 3년 전)")
    parser.add_argument("--end", default=end.isoformat(), help="종료일 YYYY-MM-DD (기본: 한국 시간 어제)")
    parser.add_argument("--profile", default="kiwoom-real", help="윈도우에 저장된 키움 계좌 별칭")
    parser.add_argument("--unadjusted", action="store_true", help="수정주가 대신 원주가 조회")
    default_output = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" / "data" if sys.platform == "win32" else Path("data")
    parser.add_argument("--output", type=Path, default=default_output, help="결과 폴더")
    parser.add_argument("--max-pages", type=int, default=100, help="최대 조회 페이지 수")
    args = parser.parse_args(argv)
    try:
        code = args.code
        if code is None:
            code = input("수집할 종목코드 [Enter: 005930 삼성전자]: ").strip() if sys.stdin.isatty() else "005930"
            code = code or "005930"
        start, end = parse_date(args.start), parse_date(args.end)
        validate_request(code, start, end)
        if args.max_pages < 1:
            raise CollectionError("최대 조회 페이지 수는 1 이상이어야 합니다.")
        print(f"종목 {code} | 요청 기간 {start} ~ {end} | {'수정주가' if not args.unadjusted else '원주가'}")
        print("키움에 저장된 인증정보로 일봉을 조회합니다.", flush=True)
        source = KiwoomSource(args.profile)
        result = collect(source, code, start, end, adjusted=not args.unadjusted, max_pages=args.max_pages,
                         progress=lambda page, rows: print(f"  {page}페이지 확인 · 기간 내 {rows}거래일", flush=True))
        target = save_collection(result, args.output)
        print(f"\n수집·저장 완료: {len(result.bars)}거래일 ({result.bars[0].date} ~ {result.bars[-1].date})")
        print(f"저장 위치: {target.resolve()}")
        print("daily.csv: 가격·거래량 / manifest.json: 수집 기록과 품질 정보")
        if result.stop_reason == "source_history_exhausted":
            print("안내: 키움의 조회 가능한 기록 끝에 도달했습니다. 실제 수집 시작일을 확인해주세요.")
        if any(b.volume_capped for b in result.bars):
            print("주의: 거래량이 상한값인 날짜가 있습니다. 해당 값은 학습 전에 확인해야 합니다.")
        print("거래일 달력과의 누락 대조, 뉴스·재무 분석, 학습·자동매매는 아직 수행하지 않았습니다.")
        return 0
    except CollectionError as exc:
        print(f"수집 실패: {exc}", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("수집을 취소했습니다. 완료 파일은 만들지 않았습니다.", file=sys.stderr)
        return 130
    except PermissionError:
        print("수집 실패: 저장 폴더 또는 키움 인증 캐시에 접근할 수 없습니다. START.cmd로 실행하고 오류를 알려주세요.", file=sys.stderr)
    except OSError:
        print("수집 실패: 네트워크 또는 파일 저장 작업을 완료하지 못했습니다. 완료 여부를 확인해주세요.", file=sys.stderr)
    except Exception:
        # SDK exceptions can contain server responses. Do not echo credentials or raw request data.
        print("수집 실패: 키움 조회를 완료하지 못했습니다. 먼저 기존 키움 연결의 doctor 결과를 확인해주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
