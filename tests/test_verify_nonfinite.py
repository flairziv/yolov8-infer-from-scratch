"""检查对拍工具是否拒绝逐位相同的 NaN/Inf；只依赖 Python 标准库和已构建的 yinfer。"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("YINFER_BIN", ROOT / "build" / "yinfer")).resolve()


class VerifyNonfiniteTest(unittest.TestCase):
    def verify_copy(self, bits):
        # Split 只复制比特，保证输出与参考完全相同，专门检查 EXACT 是否错误地绕过有限数检查。
        with tempfile.TemporaryDirectory(prefix="yinfer-verify-") as tmp:
            root = Path(tmp)
            model = root / "model"
            ref = root / "ref"
            model.mkdir()
            ref.mkdir()
            (model / "model.txt").write_text(
                "format yolov8-infer 1\n"
                "weights weights.bin 0\n"
                "input x\n"
                "output y\n"
                "tensor x 1 act\n"
                "tensor y 1 act\n"
                "node Split copy in=x out=y axis=0 sizes=1\n",
                encoding="utf-8",
            )
            (model / "weights.bin").write_bytes(b"")
            value = struct.pack("<I", bits)
            (ref / "ref.bin").write_bytes(value + b"\0" * 60 + value)
            (ref / "ref.txt").write_text("x 0 1 1\ny 64 1 1\n", encoding="utf-8")
            return subprocess.run(
                [str(BINARY), "verify", str(model), str(ref), "--brief"],
                capture_output=True, text=True, encoding="utf-8", check=False,
            )

    def test_finite_exact_value_passes(self):
        for bits in (0x00000000, 0x80000000, 0x3F800000):
            with self.subTest(bits=hex(bits)):
                result = self.verify_copy(bits)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("EXACT 1 OK 0 FAIL 0 REF 0", result.stdout)

    def test_identical_nonfinite_values_fail(self):
        for bits in (0x7FC00000, 0x7F800000, 0xFF800000):
            with self.subTest(bits=hex(bits)):
                result = self.verify_copy(bits)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("EXACT 0 OK 0 FAIL 1 REF 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
