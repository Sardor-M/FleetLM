"""The node agent's connection path is wired to methods that exist.

run() schedules coroutines by name. One that goes missing is invisible to every
other test, because nothing else reaches run().
"""

import inspect
import re

from node_agent.__main__ import NodeAgent

# ── wiring ──────────────────────────────────────────────────────────────

def test_every_method_the_connection_path_calls_exists():
    """A method referenced by run() but never defined stays invisible to tests.

    Nothing else reaches run(), so the agent imports fine, the suite passes,
    and the node dies with AttributeError the moment it connects to a fleet.
    """
    source = inspect.getsource(NodeAgent.run) + inspect.getsource(NodeAgent._message_loop)
    referenced = set(re.findall(r"self\.(_\w+)\(", source))
    missing = sorted(n for n in referenced if not hasattr(NodeAgent, n))

    assert not missing, f"run()/_message_loop call methods that do not exist: {missing}"
