"""User-operated interactive OpenDART key setup into Windows credential storage."""
import argparse
import re
import sys
from getpass import getpass

from dart_client import DartClient, DartError, KEY_NAME, SERVICE


def configure(backend, *, replace=False, prompt=getpass, client_factory=DartClient):
    existing = backend.get_password(SERVICE, KEY_NAME)
    if existing and not replace:
        client_factory(existing).validate_key()
        return "등록된 DART 인증키로 기업 조회가 성공했습니다. 기존 값을 유지합니다."
    key = prompt("DART API 인증키 (입력한 값은 화면에 표시되지 않습니다): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9]{40}", key):
        raise DartError("발급받은 DART 인증키 40자리를 입력해주세요.")
    client_factory(key).validate_key()
    backend.set_password(SERVICE, KEY_NAME, key)
    if backend.get_password(SERVICE, KEY_NAME) != key:
        raise DartError("윈도우 보안 저장소의 DART 인증키 저장을 확인하지 못했습니다.")
    return "DART 인증키 검증 및 윈도우 보안 저장소 저장을 완료했습니다."


def main(argv=None):
    parser = argparse.ArgumentParser(description="DART 인증키를 PC 보안 저장소에 등록")
    parser.add_argument("--replace", action="store_true", help="검증한 새 키로 기존 키 교체")
    args = parser.parse_args(argv)
    if sys.platform != "win32" or not sys.stdin.isatty():
        print("사용자님의 윈도우 PC에서 DART_SETUP.cmd를 직접 실행해주세요.", file=sys.stderr)
        return 1
    try:
        import keyring
        print(configure(keyring, replace=args.replace))
        return 0
    except DartError as exc:
        print(f"DART 설정 실패: {exc}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("입력을 취소했습니다.", file=sys.stderr)
    except Exception:
        print("DART 인증키를 보안 저장소에 저장하지 못했습니다. 오류 발생 단계를 알려주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
