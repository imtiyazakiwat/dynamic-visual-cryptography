"""Command-line interface.

    python -m dvc.cli run samples/peppers_color.png --k 2 --n 3 --mode xor --out out/
    python -m dvc.cli run samples/secret_text_binary.png --mode or --fault 1=flip --fault 3=drop
    python -m dvc.cli history
    python -m dvc.cli replay <session_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import DVCError
from .session import DEFAULT_WORKDIR, SessionConfig, read_log, replay, run_session, save_artifacts
from .transport import FAULT_KINDS


def _parse_faults(items: list[str]) -> dict[int, str]:
    faults = {}
    for item in items or []:
        try:
            index, kind = item.split("=", 1)
            index = int(index)
        except ValueError:
            raise SystemExit(f"--fault expects INDEX=KIND, got '{item}'")
        if kind not in FAULT_KINDS:
            raise SystemExit(f"Unknown fault '{kind}'. Choose from: {', '.join(FAULT_KINDS)}")
        faults[index] = kind
    return faults


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def cmd_run(args) -> int:
    config = SessionConfig(
        k=args.k, n=args.n, recovery=args.mode, chaos=args.chaos, force_mode=args.image_mode,
        faults=_parse_faults(args.fault), persistent_faults=args.persistent, max_retries=args.retries,
        store_seed=not args.no_seed, sharpen=args.sharpen,
        manifest_fault="tamper" if args.tamper_manifest else "none",
    )
    result = run_session(args.image, config, args.workdir)
    print(f"session   {result.session_id}")
    print(f"status    {result.status}")
    for phase, state in result.gates.items():
        timing = result.timings.get(f"{phase}_ms")
        print(f"  {phase:<12} {state:<8} {'' if timing is None else f'{timing:8.1f} ms'}")
    if result.attempts:
        for number, attempt in enumerate(result.attempts, 1):
            checks = "  ".join(f"#{c.index}:{c.status}" for c in attempt.checks)
            print(f"verify#{number}  {checks}")
    if result.error:
        print(f"error     {result.error['code']} in {result.error['phase']}: {result.error['message']}")
    if result.ok:
        print("scorecard")
        for target in result.scores["targets"]:
            verdict = {True: "PASS", False: "FAIL", None: "n/a "}[target["passed"]]
            note = f"  ({target['note']})" if target["note"] else ""
            print(f"  [{verdict}] {target['metric']:<36} {_fmt(target['value']):>10}  target {target['target']}{note}")
    if args.out:
        for path in save_artifacts(result, args.out):
            print(f"wrote     {path}")
    if args.json:
        print(json.dumps(result.record, indent=2))
    return 0 if result.ok else 2


def cmd_history(args) -> int:
    records = read_log(args.workdir)
    for record in records[-args.limit:]:
        config = record.get("config", {})
        error = (record.get("error") or {}).get("code", "")
        print(f"{record.get('session_id') or '-':<18} {record['status']:<8} k={config.get('k')} n={config.get('n')} "
              f"{config.get('recovery'):<4} {record.get('image_mode', '-'):<7} {error}")
    if not records:
        print("No sessions logged yet.")
    return 0


def cmd_replay(args) -> int:
    try:
        outcome = replay(args.session_id, args.workdir)
    except DVCError as error:
        print(f"replay failed: {error}")
        return 2
    print(f"session      {outcome['session_id']}")
    print(f"bit_exact    {outcome['bit_exact']}")
    print(f"tags         {outcome['tag_matches']}")
    print(f"output_match {outcome['output_match']}")
    print(f"replay_ms    {outcome['replay_ms']:.1f}")
    return 0 if outcome["bit_exact"] else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dvc", description="Dynamic Visual Cryptography for Secure Image Transmission")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="where channels, logs and inputs live")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one full session: encode, dispatch, verify, reconstruct, score")
    run.add_argument("image", type=Path)
    run.add_argument("--k", type=int, default=2)
    run.add_argument("--n", type=int, default=3)
    run.add_argument("--mode", choices=["xor", "or"], default="xor", help="recovery mode")
    run.add_argument("--image-mode", choices=["binary", "gray", "color"], default=None)
    run.add_argument("--chaos", action="store_true", help="xor-mix a logistic-map stream (auxiliary)")
    run.add_argument("--fault", action="append", metavar="INDEX=KIND",
                     help=f"inject a fault on a channel; KIND is one of {', '.join(FAULT_KINDS)}")
    run.add_argument("--persistent", action="store_true", help="faults also hit retransmissions")
    run.add_argument("--retries", type=int, default=1)
    run.add_argument("--tamper-manifest", action="store_true")
    run.add_argument("--no-seed", action="store_true", help="log only a seed commitment (disables replay)")
    run.add_argument("--sharpen", action="store_true", help="unsharp-mask OR outputs")
    run.add_argument("--out", type=Path, help="write secret, shares, output and manifest here")
    run.add_argument("--json", action="store_true", help="print the full session record")
    run.set_defaults(func=cmd_run)

    history = sub.add_parser("history", help="list logged sessions")
    history.add_argument("--limit", type=int, default=20)
    history.set_defaults(func=cmd_history)

    rep = sub.add_parser("replay", help="regenerate a logged session and check it bit-for-bit")
    rep.add_argument("session_id")
    rep.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
