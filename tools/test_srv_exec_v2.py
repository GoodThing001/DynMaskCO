import base64
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _srv_exec_v2 as runner


class ExactRemoteArgv(unittest.TestCase):
    def test_literal_arguments_once_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            script = p / 'file with spaces.py'
            script.write_text('import json,sys\nfrom pathlib import Path\n'
                              'Path("argv.json").write_text(json.dumps(sys.argv[1:]))\n'
                              'with Path("count").open("a") as f:f.write("1")\n'
                              'sys.exit(7)\n')
            args = ['a b', '"quoted"', "it's", '$HOME', '$(touch secret)',
                    '`echo injected`', '; exit 0', '汉字', '']
            result = subprocess.run([sys.executable, '-c', runner.bootstrap(
                folder, sys.executable, str(script), args)])
            import json
            self.assertEqual(json.loads((p / 'argv.json').read_text()), args)
            self.assertEqual((p / 'count').read_text(), '1')
            self.assertEqual(result.returncode, 7)

    def test_windows_shell_contains_only_encoded_payload(self):
        cmd = runner.build('3080.json', "C:/it's a folder", 'D:/py space/python.exe',
                           'a.py', ['$(get-secret)', '"x y"'])
        self.assertNotIn('get-secret', cmd)
        ps = base64.b64decode(cmd.split()[-1]).decode('utf-16-le')
        self.assertIn('exit $LASTEXITCODE', ps)
        self.assertNotIn('get-secret', ps)

    def test_linux_executable_quoted(self):
        cmd = runner.build('server.json', '/dir space', '/env space/python', 'a.py', [';exit'])
        self.assertTrue(cmd.startswith("'/env space/python' -c "))

    def test_cli_propagates_remote_error(self):
        from unittest.mock import patch
        with patch.object(runner, 'run_capture', return_value={
                'stdout': '', 'stderr': 'expected fixture failure\n', 'exit_code': 23}):
            self.assertEqual(runner.main(['--workdir', '/', '--python', 'py', '--script', 'x']), 23)


if __name__ == '__main__':
    unittest.main()
