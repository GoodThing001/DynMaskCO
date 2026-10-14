"""Compatibility entry point for exact-argv SSH execution.

Arguments are JSON/base64 transported by v2; remote nonzero exit codes propagate.
Use foreground mode only for bounded checks. Linux runs exceeding five minutes
belong in their registered tmux pipeline, not a local background SSH connection.
The prior version is retained in the independent audit source snapshot.
"""
from _srv_exec_v2 import build, main


if __name__ == '__main__':
    raise SystemExit(main())
