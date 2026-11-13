
import time
import threading

from collections import deque
from datetime import datetime, timezone, date


# ==========================================
# RATE LIMIT ERROR
# ==========================================

class RateLimitExceeded(Exception):
    """Simulated HTTP 429 rate-limit response."""

    def __init__(
        self,
        message,
        limit_type,
        retry_after=0
    ):

        super().__init__(message)

        self.status_code = 429
        self.limit_type = limit_type
        self.retry_after = max(0, retry_after)


# ==========================================
# RATE LIMITER
# ==========================================

class RateLimiter:

    def __init__(
        self,
        rpm_limit=30,
        daily_limit=1000,
        window_seconds=60
    ):

        if rpm_limit < 1:
            raise ValueError("RPM limit must be at least 1.")

        if daily_limit < 1:
            raise ValueError("Daily limit must be at least 1.")

        if window_seconds <= 0:
            raise ValueError("Window must be positive.")

        self.rpm_limit = rpm_limit
        self.daily_limit = daily_limit
        self.window_seconds = window_seconds

        self.request_timestamps = deque()

        self.daily_requests = 0
        self.current_date = self._today_utc()

        self.lock = threading.Lock()

        self.accepted_requests = 0
        self.rejected_rpm = 0
        self.rejected_daily = 0

    # ======================================
    # CURRENT UTC DATE
    # ======================================

    @staticmethod
    def _today_utc():

        return datetime.now(
            timezone.utc
        ).date()

    # ======================================
    # RESET DAILY COUNTER
    # ======================================

    def _refresh_daily_counter(self):

        today = self._today_utc()

        if today != self.current_date:

            self.current_date = today
            self.daily_requests = 0

    # ======================================
    # REMOVE EXPIRED REQUESTS
    # ======================================

    def _remove_expired_requests(self, now):

        cutoff = now - self.window_seconds

        while (
            self.request_timestamps
            and self.request_timestamps[0] <= cutoff
        ):

            self.request_timestamps.popleft()

    # ======================================
    # CHECK AND RECORD REQUEST
    # ======================================

    def acquire(self):

        with self.lock:

            now = time.monotonic()

            self._refresh_daily_counter()

            self._remove_expired_requests(now)

            # Check daily request budget first.
            if self.daily_requests >= self.daily_limit:

                self.rejected_daily += 1

                raise RateLimitExceeded(
                    "Daily request budget exhausted.",
                    limit_type="daily",
                    retry_after=0
                )

            # Check rolling requests-per-minute limit.
            if len(self.request_timestamps) >= self.rpm_limit:

                oldest_request = self.request_timestamps[0]

                retry_after = (
                    oldest_request
                    + self.window_seconds
                    - now
                )

                self.rejected_rpm += 1

                raise RateLimitExceeded(
                    "Requests-per-minute limit exceeded.",
                    limit_type="rpm",
                    retry_after=retry_after
                )

            # Accept and record request.
            self.request_timestamps.append(now)

            self.daily_requests += 1
            self.accepted_requests += 1

            return {
                "accepted": True,
                "requests_in_window": len(
                    self.request_timestamps
                ),
                "daily_requests": self.daily_requests,
                "rpm_limit": self.rpm_limit,
                "daily_limit": self.daily_limit
            }

    # ======================================
    # CURRENT LIMIT STATUS
    # ======================================

    def get_stats(self):

        with self.lock:

            now = time.monotonic()

            self._refresh_daily_counter()

            self._remove_expired_requests(now)

            return {
                "rpm_limit": self.rpm_limit,
                "daily_limit": self.daily_limit,
                "requests_in_current_window": len(
                    self.request_timestamps
                ),
                "daily_requests": self.daily_requests,
                "daily_remaining": max(
                    0,
                    self.daily_limit - self.daily_requests
                ),
                "accepted_requests": self.accepted_requests,
                "rejected_rpm": self.rejected_rpm,
                "rejected_daily": self.rejected_daily
            }


# ==========================================
# SMOKE TEST
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("THE RELAY - RATE LIMITER")
    print("=" * 60)

    limiter = RateLimiter(
        rpm_limit=3,
        daily_limit=5,
        window_seconds=60
    )

    # Test requests-per-minute limit.
    print("\nTEST 1: Requests-per-minute limit")

    for request_number in range(1, 5):

        try:

            result = limiter.acquire()

            print(
                f"Request {request_number}: ACCEPTED | "
                f"{result}"
            )

        except RateLimitExceeded as error:

            print(
                f"Request {request_number}: HTTP "
                f"{error.status_code} | "
                f"Type: {error.limit_type} | "
                f"Retry after: {error.retry_after:.2f}s"
            )

    print("\nCurrent statistics:")
    print(limiter.get_stats())

    # Test daily budget using a fresh limiter.
    print("\nTEST 2: Daily request budget")

    daily_limiter = RateLimiter(
        rpm_limit=10,
        daily_limit=2,
        window_seconds=60
    )

    for request_number in range(1, 4):

        try:

            daily_limiter.acquire()

            print(
                f"Daily request {request_number}: ACCEPTED"
            )

        except RateLimitExceeded as error:

            print(
                f"Daily request {request_number}: "
                f"HTTP {error.status_code} | "
                f"Type: {error.limit_type}"
            )

    print("\nDaily limiter statistics:")
    print(daily_limiter.get_stats())
