import asyncio
import time

from edb.providers.base import RateLimiter


def test_rate_limiter_spaces_requests():
    async def go():
        limiter = RateLimiter(per_minute=600)  # 0.1 s apart
        t0 = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(4)))
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert 0.28 <= elapsed < 0.6
