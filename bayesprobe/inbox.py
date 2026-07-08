from __future__ import annotations

from bayesprobe.schemas import ExternalSignal, SignalInboxStatus


class SignalInbox:
    def __init__(self, cycle_id: str):
        self.cycle_id = cycle_id
        self._signals: list[ExternalSignal] = []
        self._deferred_signals: list[ExternalSignal] = []
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def deferred_signals(self) -> list[ExternalSignal]:
        return list(self._deferred_signals)

    def add(self, signal: ExternalSignal) -> ExternalSignal:
        if self._closed:
            deferred = signal.model_copy(update={"inbox_status": SignalInboxStatus.DEFERRED})
            self._deferred_signals.append(deferred)
            return deferred
        accepted = signal.model_copy(
            update={
                "cycle_id": self.cycle_id,
                "inbox_status": SignalInboxStatus.ACCEPTED,
            }
        )
        self._signals.append(accepted)
        return accepted

    def close(self) -> list[ExternalSignal]:
        self._closed = True
        return list(self._signals)
