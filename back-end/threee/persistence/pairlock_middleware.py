import logging
from datetime import datetime, timezone
from typing import List, Optional

from threee.exchange import timeframe_to_next_date
from threee.persistence.models import PairLock


class PairLocks():

    use_db = True
    locks: List[PairLock] = []

    timeframe: str = ''

    @staticmethod
    def reset_locks() -> None:
        if not PairLocks.use_db:
            PairLocks.locks = []

    @staticmethod
    def lock_pair(pair: str, until: datetime, reason: str = None, *,
                  now: datetime = None) -> PairLock:
        lock = PairLock(
            pair=pair,
            lock_time=now or datetime.now(timezone.utc),
            lock_end_time=timeframe_to_next_date(PairLocks.timeframe, until),
            reason=reason,
            active=True
        )
        if PairLocks.use_db:
            PairLock.query.session.add(lock)
            PairLock.query.session.commit()
        else:
            PairLocks.locks.append(lock)
        return lock

    @staticmethod
    def get_pair_locks(pair: Optional[str], now: Optional[datetime] = None) -> List[PairLock]:
        if not now:
            now = datetime.now(timezone.utc)

        if PairLocks.use_db:
            return PairLock.query_pair_locks(pair, now).all()
        else:
            locks = [lock for lock in PairLocks.locks if (
                lock.lock_end_time >= now
                and lock.active is True
                and (pair is None or lock.pair == pair)
            )]
            return locks

    @staticmethod
    def get_pair_longest_lock(pair: str, now: Optional[datetime] = None) -> Optional[PairLock]:
        locks = PairLocks.get_pair_locks(pair, now)
        locks = sorted(locks, key=lambda l: l.lock_end_time, reverse=True)
        return locks[0] if locks else None

    @staticmethod
    def unlock_pair(pair: str, now: Optional[datetime] = None) -> None:
        if not now:
            now = datetime.now(timezone.utc)

        locks = PairLocks.get_pair_locks(pair, now)
        for lock in locks:
            lock.active = False
        if PairLocks.use_db:
            PairLock.query.session.commit()

    @staticmethod
    def unlock_reason(reason: str, now: Optional[datetime] = None) -> None:
        if not now:
            now = datetime.now(timezone.utc)

        if PairLocks.use_db:

            filters = [PairLock.lock_end_time > now,
                       PairLock.active.is_(True),
                       PairLock.reason == reason
                       ]
            locks = PairLock.query.filter(*filters)
            for lock in locks:
                lock.active = False
            PairLock.query.session.commit()
        else:
            locks = PairLocks.get_pair_locks(None)
            for lock in locks:
                if lock.reason == reason:
                    lock.active = False

    @staticmethod
    def is_global_lock(now: Optional[datetime] = None) -> bool:
        if not now:
            now = datetime.now(timezone.utc)

        return len(PairLocks.get_pair_locks('*', now)) > 0

    @staticmethod
    def is_pair_locked(pair: str, now: Optional[datetime] = None) -> bool:
        if not now:
            now = datetime.now(timezone.utc)

        return len(PairLocks.get_pair_locks(pair, now)) > 0 or PairLocks.is_global_lock(now)

    @staticmethod
    def get_all_locks() -> List[PairLock]:
        if PairLocks.use_db:
            return PairLock.query.all()
        else:
            return PairLocks.locks
