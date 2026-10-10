"""Separate process: deliberately bypass all cleanup at a durable phase."""

import os
import sys

from backend_support import request, runtime, services

root, uri, collection, point, operation = sys.argv[1:]


def fault(phase):
    if phase == point:
        os._exit(86)


backend = runtime(root, uri, collection, fault=fault)
backend.start()
if operation in {"add", "add_many"}:
    body = request()
    if operation == "add_many":
        body = body.model_copy(update={"messages": body.messages * 129})
    services(backend)[0].handle(body)
else:
    backend.purge("alice")
raise SystemExit("fault point was not reached")
