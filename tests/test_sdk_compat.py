import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kiwoom.core import platform_paths

from lyupang import KiwoomSource
from sdk_compat import _ACLSubprocess, configure_windows_acl_capture


class WindowsSDKCaptureTests(unittest.TestCase):
    def test_real_korean_bytes_reproduce_utf8_failure_and_are_captured_unchanged(self):
        out = "모든 파일을 처리했습니다.".encode("cp949")
        err = "추가 확인 내용".encode("cp949")
        child = [sys.executable, "-c", f"import sys;sys.stdout.buffer.write({out!r});sys.stderr.buffer.write({err!r})"]
        with self.assertRaises(UnicodeDecodeError):
            subprocess.run(child, capture_output=True, text=True, encoding="utf-8", check=False)
        calls = []
        def native_stub(command, **options):
            calls.append((command, options))
            # Use a real subprocess and bytes on both pipes, without running an
            # actual Windows ACL command or touching any credentials.
            return subprocess.run(child, **options)
        facade = _ACLSubprocess(SimpleNamespace(run=native_stub))
        command = [r"C:\Windows\System32\icacls.exe", "test-only-path", "/inheritance:r"]
        result = facade.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual((result.stdout, result.stderr), (out, err))
        self.assertEqual(calls[0][0], command)
        self.assertIs(calls[0][1]["text"], False)
        self.assertIs(calls[0][1]["check"], False)

    def test_real_sdk_acl_function_preserves_commands_and_strict_failure(self):
        commands = []
        fail = [False]
        def run(command, **options):
            commands.append((command, options))
            self.assertIs(options["text"], False)
            return subprocess.CompletedProcess(command, 1 if fail[0] else 0,
                                               "처리 결과".encode("cp949"), "진단 내용".encode("cp949"))
        delegate = SimpleNamespace(run=run)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.txt"
            path.write_text("test fixture")
            with patch.object(platform_paths, "subprocess", _ACLSubprocess(delegate)), \
                 patch.object(platform_paths, "os", SimpleNamespace(name="nt", environ={"USERNAME": r"pc\pc"})), \
                 patch.object(platform_paths.shutil, "which", return_value=r"C:\Windows\System32\icacls.exe"):
                self.assertEqual(platform_paths.protect_file(path, strict=True), path)
                self.assertEqual(commands[0][0][1:], [str(path), "/inheritance:r"])
                self.assertEqual(commands[1][0][1:], [str(path), "/grant:r", r"pc\pc:F"])
                commands.clear()
                platform_paths.ensure_private_directory(Path(tmp), strict=True)
                self.assertEqual(commands[1][0][-2:], ["/t", "/c"])
                fail[0] = True
                with self.assertRaises(PermissionError):
                    platform_paths.protect_file(path, strict=True)
                self.assertEqual(path.read_text(), "test fixture")

    def test_only_module_local_icacls_capture_changes_and_installation_is_idempotent(self):
        standard_run = subprocess.run
        delegate = Mock()
        with patch("sdk_compat.sys", SimpleNamespace(platform="win32")), patch.object(platform_paths, "subprocess", delegate):
            self.assertTrue(configure_windows_acl_capture())
            first = platform_paths.subprocess
            self.assertIsInstance(first, _ACLSubprocess)
            self.assertTrue(configure_windows_acl_capture())
            self.assertIs(platform_paths.subprocess, first)
            self.assertIs(first._delegate, delegate)
            self.assertIs(subprocess.run, standard_run)
        with patch("sdk_compat.sys", SimpleNamespace(platform="linux")), patch.object(platform_paths, "subprocess", delegate):
            self.assertFalse(configure_windows_acl_capture())
            self.assertIs(platform_paths.subprocess, delegate)

    def test_unrelated_commands_explicit_encoding_and_binary_mode_pass_through(self):
        delegate = Mock()
        facade = _ACLSubprocess(delegate)
        cases = [
            (["other.exe", "data"], {"capture_output": True, "text": True}),
            (["icacls.exe", "data"], {"capture_output": True, "text": True, "encoding": "cp949"}),
            (["icacls.exe", "data"], {"capture_output": True, "text": False}),
            ("icacls data", {"capture_output": True, "text": True}),
        ]
        for command, options in cases:
            with self.subTest(command=command, options=options):
                self.assertIs(facade.run(command, **options), delegate.run.return_value)
                delegate.run.assert_called_with(command, **options)

    def test_client_factory_is_called_after_windows_compatibility_is_installed(self):
        delegate = Mock()
        def make_client(**options):
            self.assertIsInstance(platform_paths.subprocess, _ACLSubprocess)
            self.assertEqual(options, {"profile": "kiwoom-real", "timeout_seconds": 30})
            return client
        client = object()
        with patch("sdk_compat.sys", SimpleNamespace(platform="win32")), \
             patch.object(platform_paths, "subprocess", delegate), patch("kiwoom.get_client", side_effect=make_client):
            source = KiwoomSource()
        self.assertIs(source._client, client)

    def test_captured_permission_errors_are_not_printed_or_changed_into_success(self):
        delegate = Mock()
        delegate.run.return_value = subprocess.CompletedProcess([], 5, b"PRIVATE_DIAGNOSTIC", b"PRIVATE_DIAGNOSTIC")
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(platform_paths, "subprocess", _ACLSubprocess(delegate)), \
             patch.object(platform_paths.shutil, "which", return_value="icacls.exe"), \
             redirect_stdout(output), redirect_stderr(errors):
            self.assertFalse(platform_paths._protect_windows_path(Path("fixture"), is_dir=False))
        self.assertEqual(output.getvalue() + errors.getvalue(), "")
        self.assertEqual(delegate.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
