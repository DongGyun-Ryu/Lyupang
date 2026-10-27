"""Build a Windows source bundle; credentials, data and venvs are never included."""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("collect.py", "lyupang.py", "START.cmd", "OPEN_DATA.cmd", "README.md", "research.py", "research_data.py",
         "dart_client.py", "setup_dart.py", "RESEARCH.cmd", "DART_SETUP.cmd")


def build():
    target = ROOT / "dist" / "Lyupang-data-0.2.0.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES:
            payload = (ROOT / name).read_bytes()
            if name.endswith(".cmd"):
                payload = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            bundle.writestr("Lyupang-data/" + name, payload)
        guide = """Lyupang 국내 주식 자료·사례 준비 0.2.0

1. ZIP 파일을 '모두 추출'합니다.
2. START.cmd를 실행합니다.
3. 종목코드 질문에서 Enter를 누르면 삼성전자(005930)를 수집합니다.
4. '수집·저장 완료'와 거래일 수가 나오는지 확인합니다.
5. OPEN_DATA.cmd를 실행하면 결과 폴더가 열립니다.

여러 종목과 비교 사례:
RESEARCH.cmd를 실행하면 예시 종목 3개를 준비하고 브라우저 보고서를 엽니다.
기존 자료가 같은 기간·수정주가 기준이고 무결성 검사를 통과하면 재사용합니다.
DART 키가 없으면 재무·공시는 대기로 표시하고 가격 비교 사례만 만듭니다.

재무·공시 연결:
OpenDART(https://opendart.fss.or.kr/)에서 인증키를 발급받은 뒤
DART_SETUP.cmd를 실행해 그 창에 직접 입력하세요. 채팅에 보내지 마세요.
기존 키움 인증키와는 별개이며 윈도우 보안 저장소에 저장합니다.
등록 후 RESEARCH.cmd를 다시 실행하면 재무·공시를 연결합니다.

기본 수집 기간은 약 3년 전부터 한국 시간 기준 어제까지입니다.
기존 키움 인증정보를 사용하므로 키를 새로 입력하지 않습니다.
결과: daily.csv(가격·거래량), manifest.json(수집 기록·품질 정보)
저장 위치: %LOCALAPPDATA%\\Lyupang\\data

오류가 나오면 실행 창의 문구를 알려주세요. 인증키는 보내지 마세요.
계정 이름 보정은 START.cmd가 자동으로 적용합니다.
이 버전은 조회·저장·사례 준비만 수행합니다. 자동매매·모델 학습은 없습니다.
기존 단일 종목 수집은 사용자 PC에서 검증됐습니다.
새 여러 종목·DART 기능의 실제 결과는 PC 실행 후 확인해야 합니다.

상세한 기능과 데이터 한계는 README.md에 있습니다.
"""
        bundle.writestr("Lyupang-data/먼저읽어주세요.txt", guide.encode("utf-8-sig"))
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(checksum + "  " + target.name + "\n", encoding="ascii")
    print(target)
    print("SHA256:", checksum)
    return target


if __name__ == "__main__":
    build()
