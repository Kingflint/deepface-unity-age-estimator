"""Admin command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import load_settings
from .openapi import generate_openapi_spec
from .storage import get_connection
from .storage.migrations import latest_version, migrate


def cmd_version(_args: argparse.Namespace) -> int:
    from . import __version__

    print(__version__)
    return 0


def cmd_config(_args: argparse.Namespace) -> int:
    settings = load_settings()
    print(
        json.dumps(
            {
                "port": settings.port,
                "log_level": settings.log_level,
                "debug": settings.debug,
                "max_image_bytes": getattr(settings, "max_image_bytes", None),
                "actions": list(getattr(settings, "actions", ()) or []),
            },
            indent=2,
        )
    )
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    factory = get_connection(args.database)
    try:
        applied = migrate(factory)
    finally:
        factory.close()
    print(
        json.dumps(
            {
                "database": args.database,
                "applied": list(applied),
                "latest_known": latest_version(),
            },
            indent=2,
        )
    )
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    spec = generate_openapi_spec()
    out = json.dumps(spec, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        print(out)
    return 0


def cmd_smoke(_args: argparse.Namespace) -> int:
    from . import create_app

    app = create_app()
    with app.test_client() as client:
        for path in ("/", "/healthz"):
            resp = client.get(path)
            print(path, "->", resp.status_code)
            if resp.status_code >= 500:
                return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepface-server",
        description="Operational CLI for deepface-unity-age-estimator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print package version").set_defaults(func=cmd_version)
    sub.add_parser("config", help="Dump effective configuration").set_defaults(func=cmd_config)

    migrate_parser = sub.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument("--database", default=":memory:")
    migrate_parser.set_defaults(func=cmd_migrate)

    openapi_parser = sub.add_parser("openapi", help="Generate OpenAPI specification")
    openapi_parser.add_argument("--output", default=None)
    openapi_parser.add_argument("--pretty", action="store_true")
    openapi_parser.set_defaults(func=cmd_openapi)

    sub.add_parser("smoke", help="Run a basic smoke test").set_defaults(func=cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
