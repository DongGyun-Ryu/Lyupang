"""Create/resume a frozen manual text review, or compare downloaded answers."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

from lyupang import CollectionError, SEOUL
from text_validation import (RUBRIC, canonical, digest, evaluate, evaluation_html,
                             latest_context, load_pack, make_pack, read_json)


def prepare(source, root):
    # Validate the current source before reusing an earlier matching review.
    from text_validation import source_rows
    summary, _, _, _ = source_rows(source)
    pack_id = digest((summary["statement_sha256"] + ":" + RUBRIC).encode())[:24]
    parent = root / "text_validation" / "packs"
    target = parent / pack_id
    if target.exists():
        manifest, _, _ = load_pack(target)
        if manifest["source_statements_sha256"] != summary["statement_sha256"]:
            raise CollectionError("기존 검토 묶음과 현재 분류 결과가 다릅니다.")
        if not (target / "review.html").is_file():
            raise CollectionError("기존 검토 화면 파일을 찾지 못했습니다.")
        return target
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        make_pack(source, staging / "pack")
        # Do not replace another process's completed pack.
        os.rename(staging / "pack", target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def downloaded_labels(pack, folders):
    manifest, _, _ = load_pack(pack)
    candidates = []
    for folder in folders:
        if not folder.is_dir():
            continue
        # Browser duplicate names '(1)' etc are accepted. Unrelated files ignored.
        for path in folder.glob(f"Lyupang-text-labels-{manifest['pack_id']}*.json"):
            if path.is_file() and not path.is_symlink():
                candidates.append(path)
    if not candidates:
        return None
    # Inspect the newest matching filename. A corrupt/latest file must not be
    # silently replaced with an older favorable answer set.
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def record_evaluation(pack, labels, root):
    original_labels = labels.read_bytes()
    summary, comparisons = evaluate(pack, labels)
    if labels.read_bytes() != original_labels or digest(original_labels) != summary["label_sha256"]:
        raise CollectionError("비교 중 답안 파일이 바뀌었습니다. 저장을 마친 뒤 다시 실행해주세요.")
    _, items, _ = load_pack(pack)
    parent = root / "text_validation" / "results"
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    target = parent / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        summary["source_pack"] = str(pack.resolve())
        (staging / "summary.json").write_bytes(canonical(summary))
        (staging / "comparisons.json").write_bytes(canonical(comparisons))
        (staging / "reviewer_answers.json").write_bytes(original_labels)
        (staging / "report.html").write_text(evaluation_html(summary, comparisons, items), encoding="utf-8")
        # Freeze review questions/predictions alongside answers for later audit.
        for name in ("manifest.json", "items.json", "predictions.json"):
            shutil.copyfile(pack / name, staging / name)
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, summary


def main(argv=None):
    default_root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="문장 직접 검토와 고정 분류 비교")
    parser.add_argument("labels", nargs="?", type=Path, help="다운로드한 검토 답안 JSON; 생략하면 다운로드 폴더 확인")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--input", type=Path, help="기존 CONTEXT 실행 폴더")
    parser.add_argument("--pack", type=Path, help="기존 검토 묶음; 다른 분류 결과 생성 뒤 이전 답을 비교할 때")
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--review", action="store_true", help="답안 파일이 있어도 문장 검토 화면 열기")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.pack:
            pack = args.pack
        elif args.labels:
            header = read_json(args.labels, 2_000_000)
            old_id = header.get("pack_id", "")
            if not isinstance(old_id, str) or len(old_id) != 24 or any(char not in "0123456789abcdef" for char in old_id):
                raise CollectionError("답안의 검토 묶음 번호가 올바르지 않습니다.")
            pack = args.root / "text_validation" / "packs" / old_id
        else:
            pack = prepare(args.input or latest_context(args.root / "context"), args.root)
        manifest, _, _ = load_pack(pack)
        labels = args.labels or (None if args.review else downloaded_labels(pack, (args.downloads, pack)))
        if labels:
            target, summary = record_evaluation(pack, labels, args.root)
            print(f"문장 검토 비교 완료 · 답변 {summary['answered']} · 판단 유보 {summary['abstained']} · 미답변 {summary['unanswered']}")
            print("검토자 답과의 일치율입니다. 전체 정확도 검증이나 모델 승인이 아닙니다.")
            report = target / "report.html"
        else:
            print(f"직접 확인할 문장: {manifest['sample_size']}개 · 아직 사람의 답을 입력하지 않았습니다.")
            print("문장을 읽고 두 항목을 고르세요. 모르면 '판단하기 어렵습니다'를 선택해도 됩니다.")
            print("몇 문장만 답해도 됩니다. '답안 파일 저장' 후 TEXT_CHECK.cmd를 다시 실행하세요.")
            print("다운로드 폴더에서 답안을 찾지 못하면 JSON 파일을 TEXT_CHECK.cmd 위에 끌어다 놓으세요.")
            report = pack / "review.html"
        print(f"검토 화면: {report}")
        print("기존 모델·예측 기록은 유지합니다. 재학습과 주문은 실행하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open(report.resolve().as_uri())
        return 0
    except CollectionError as exc:
        print(f"문장 검토 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError:
        print("문장 분류·검토 파일을 찾지 못했습니다. 기존 CONTEXT 결과와 답안 파일 위치를 확인해주세요.", file=sys.stderr)
    except PermissionError:
        print("문장 분류·검토 폴더를 읽거나 저장할 권한을 확인해주세요.", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("문장 검토를 완료하지 못했습니다. 실행 창의 마지막 메시지를 알려주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
