"""A2A protocol — minimal message bus connecting persona-grounded agents.

The Bus is the implementation of the A2A wire layer described in the
thesis architecture chapter. It is a simple synchronous dispatcher
rather than an async transport: messages are routed in-process by
recipient name, agents register a name → callback mapping, and the
Bus enforces the broadcast / targeted distinction defined by
``Message.is_broadcast``.

The protocol decisions implemented here:

* **Routing**: explicit recipient list per message, or the single
  wildcard ``"*"`` for broadcast. There is no implicit reply-all.
* **Per-agent memory**: each agent receives only messages where it is
  the sender, listed as a recipient, or the message is a broadcast.
  This is the "independent contexts" property the multi-agent
  condition is designed to test.
* **Self-delivery**: a message is also delivered to its sender. This
  lets an agent that constructs a message via the Bus update its own
  history through the same code path used for incoming messages,
  rather than maintaining two paths. (Runner-emitted seed messages
  bypass the Bus, which is fine — the runner records them on the
  Transcript directly and tells the receiving agents to mirror them.)
* **No queueing / no async**: dispatch is synchronous; if an agent's
  receive callback raises, the run fails loudly rather than silently
  dropping the message. The pilot does not need backpressure.
"""

from __future__ import annotations

from typing import Callable

from orggraph.simulation.transcript import BROADCAST, Message

# Type alias for a per-agent receive callback.
ReceiveCallback = Callable[[Message], None]


class Bus:
    """In-process synchronous message bus for the A2A protocol.

    Use ``register(name, callback)`` to attach an agent and
    ``dispatch(message)`` to deliver. Unknown recipient names raise
    ``KeyError`` immediately so misspellings surface as bugs rather
    than silently lost messages.
    """

    def __init__(self) -> None:
        self._agents: dict[str, ReceiveCallback] = {}

    # --- registration ---------------------------------------------

    def register(self, name: str, callback: ReceiveCallback) -> None:
        """Attach an agent. Names must be unique on a given Bus."""
        if name in self._agents:
            raise ValueError(f"Agent {name!r} already registered on this Bus")
        self._agents[name] = callback

    def is_registered(self, name: str) -> bool:
        return name in self._agents

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Snapshot of registered agent names (insertion order)."""
        return tuple(self._agents)

    # --- dispatch -------------------------------------------------

    def dispatch(self, message: Message) -> tuple[str, ...]:
        """Deliver a message according to its recipient list.

        Returns the tuple of agent names the message was actually
        delivered to (the sender is included since self-delivery is
        part of the protocol — see module docstring).

        Raises:
            KeyError: a named recipient is not registered on this Bus.
            KeyError: the sender is not registered.
        """
        if message.sender not in self._agents:
            raise KeyError(
                f"Message sender {message.sender!r} is not registered on this Bus"
            )

        if message.is_broadcast():
            recipients = tuple(self._agents)
        else:
            for r in message.recipients:
                if r == BROADCAST:
                    raise ValueError(
                        f"Mixed targeted+broadcast recipient list: {message.recipients!r}"
                    )
                if r not in self._agents:
                    raise KeyError(
                        f"Recipient {r!r} not registered on this Bus "
                        f"(known: {sorted(self._agents)})"
                    )
            recipients = tuple(message.recipients)
            # Ensure the sender also sees their own message
            if message.sender not in recipients:
                recipients = recipients + (message.sender,)

        delivered: list[str] = []
        for name in recipients:
            self._agents[name](message)
            delivered.append(name)
        return tuple(delivered)
