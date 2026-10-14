"""Run one remote Python process with exact argv and propagate its exit code.

The JSON/base64 transport prevents paths and arguments from becoming shell code.
Foreground mode is for bounded checks only; long Linux work must use a registered
tmux pipeline. Credentials stay in the existing local server configuration.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import sys
import time

from _srv import _cfg


def bootstrap(workdir, python, script, args):
    payload = base64.b64encode(json.dumps(
        [workdir, python, script, list(args)], ensure_ascii=False).encode()).decode()
    return ("import base64,json,os,subprocess,sys;"
            "d=json.loads(base64.b64decode('%s'));"
            "os.chdir(d[0]);sys.exit(subprocess.run([d[1],d[2]]+d[3]).returncode)" % payload)


def build(config, workdir, python, script, args, platform=None):
    platform = platform or ('windows' if str(config).endswith('3080.json') else 'linux')
    code = bootstrap(workdir, python, script, args)
    if platform == 'linux':
        return '%s -c %s' % (shlex.quote(python), shlex.quote(code))
    if platform != 'windows':
        raise ValueError('Specify windows or linux')
    # Avoid Windows PowerShell's native argv quote conversion: the -c argument
    # below has no whitespace or double quotes, and decodes the complete program.
    encoded = base64.b64encode(code.encode()).decode()
    expression = "exec(__import__('base64').b64decode('%s'))" % encoded
    ps = "& '%s' '-c' '%s'; exit $LASTEXITCODE" % (
        python.replace("'", "''"), expression.replace("'", "''"))
    return 'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + base64.b64encode(
        ps.encode('utf-16-le')).decode()


def run_capture(command, timeout=60, config=None):
    import paramiko
    host, user, password = _cfg(config)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=22, username=user, password=password, timeout=30)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        stdin.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            out = pool.submit(stdout.read)
            err = pool.submit(stderr.read)
            result = {'stdout': out.result().decode('utf-8', 'replace'),
                      'stderr': err.result().decode('utf-8', 'replace'),
                      'exit_code': stdout.channel.recv_exit_status()}
        return result
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='.vscode/sftp.3080.json')
    parser.add_argument('--platform', choices=['windows', 'linux'])
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--timeout', type=int, default=120)
    known, rest = parser.parse_known_args(argv)
    if rest and rest[0] == '--':
        rest = rest[1:]
    start = time.perf_counter()
    result = run_capture(build(known.config, known.workdir, known.python,
                               known.script, rest, known.platform), known.timeout, known.config)
    sys.stdout.write(result['stdout'])
    sys.stderr.write(result['stderr'])
    print('[remote exit %d; local wall %.1fs]' % (
        result['exit_code'], time.perf_counter() - start), file=sys.stderr)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
