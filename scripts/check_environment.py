from __future__ import annotations

import json

from umirage.provenance import environment_manifest

if __name__ == "__main__":
    print(json.dumps(environment_manifest(), indent=2, sort_keys=True))
