"""Build a Windows source bundle; credentials, data and venvs are never included."""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("collect.py", "lyupang.py", "START.cmd", "OPEN_DATA.cmd", "README.md", "research.py", "research_data.py",
         "dart_client.py", "setup_dart.py", "RESEARCH.cmd", "DART_SETUP.cmd", "train.py", "financial_history.py",
         "learning.py", "TRAIN.cmd", "review.py", "REVIEW.cmd", "observe.py", "OBSERVE.cmd")


def build():
    target = ROOT / "dist" / "Lyupang-data-0.5.0.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES:
            payload = (ROOT / name).read_bytes()
            if name.endswith(".cmd"):
                payload = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            bundle.writestr("Lyupang-data/" + name, payload)
        guide = """Lyupang 새 예측 관찰 0.5.0

이미 학습과 진단을 마친 경우:
OBSERVE.cmd를 실행합니다. 기존 모델을 고정하고 새 예측을 기록합니다.
탈락 모델은 제외합니다. 모델 재학습과 실제 주문은 수행하지 않습니다.
처음 실행하면 브라우저에서 관찰 보고서를 엽니다.
창과 컴퓨터를 켜두면 한국 시간 매일 오전 8시 30분에 자동 확인합니다.
오전 9시 이후에는 새 예측은 대기하고 기존 결과만 비교합니다.
예측과 결과는 %LOCALAPPDATA%\\Lyupang\\forward에 보관합니다.
창을 닫거나 PC를 껐다면 다음 실행에서 이어갑니다. 놓친 예측은 소급하지 않습니다.
새 검증에도 가격·DART 조회가 필요하며 기존 인증정보를 재사용합니다.
5일봉 등의 관찰 기간이 지나야 결과가 나오므로 처음에는 결과 대기가 정상입니다.
이 단계는 예측 오차 관찰입니다. 매매 규칙·모의매매·자동 재학습은 아직 없습니다.

이미 학습을 마친 경우:
REVIEW.cmd를 실행합니다. 저장된 최신 모델과 평가 자료만 읽어 진단합니다.
재학습·키움/DART 조회·인증키 입력 없이 종목별·분기별 성능을 확인합니다.
기준보다 못한 모델은 탈락, 개선돼도 새로운 검증 전에는 보류합니다.
어떤 모델도 자동으로 실전 승인하지 않으며 실제 주문은 실행하지 않습니다.
진단 결과는 %LOCALAPPDATA%\\Lyupang\\diagnostics에 저장됩니다.
열람한 평가 기간은 evaluation_registry에 기록하며 새로운 미사용 검증으로 주장하지 않습니다.

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

기존 RESEARCH 결과로 첫 연구용 모델 학습:
TRAIN.cmd를 실행합니다. 저장한 최신 연구 결과와 기존 DART 키를 재사용합니다.
접수번호별 연간 재무 원문을 확인하고 당시 공개된 정보만 사례에 연결합니다.
조건을 만족하면 학습·검증·최종 평가 기간을 나눠 연구용 모델을 학습합니다.
학습 후 성능 진단도 자동 실행하고 판정 결과를 별도 보고서에 표시합니다.
원문을 해석하지 못하거나 표본이 부족하면 학습 대기 이유를 보고서에 표시합니다.
결과 보고서는 브라우저에서 열리고 %LOCALAPPDATA%\\Lyupang\\learning에 저장됩니다.
처음에는 원문 다운로드에 시간이 걸립니다. 같은 원문은 해시를 확인해 재사용합니다.

기본 수집 기간은 약 3년 전부터 한국 시간 기준 어제까지입니다.
기존 키움 인증정보를 사용하므로 키를 새로 입력하지 않습니다.
결과: daily.csv(가격·거래량), manifest.json(수집 기록·품질 정보)
저장 위치: %LOCALAPPDATA%\\Lyupang\\data

오류가 나오면 실행 창의 문구를 알려주세요. 인증키는 보내지 마세요.
계정 이름 보정은 START.cmd가 자동으로 적용합니다.
이 버전은 연구용 학습까지 수행할 수 있습니다. 실제 주문은 실행하지 않습니다.
기존 3종목 가격·DART 수집은 사용자 PC에서 검증됐습니다.
기존 0.3.1 원문 해석·학습은 사용자 PC에서 성공했습니다.
기존 0.4.0 진단도 사용자 PC에서 재계산이 확인됐습니다.
새 예측 기록 기능의 실제 연결은 PC 실행 후 확인해야 합니다.

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
