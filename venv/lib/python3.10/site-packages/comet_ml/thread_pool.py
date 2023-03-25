# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2022 Comet ML INC
#  This file can not be copied and/or distributed
#  without the express permission of Comet ML Inc.
# *******************************************************

import concurrent.futures
import logging
import multiprocessing
from concurrent.futures import CancelledError
from multiprocessing.pool import AsyncResult
from typing import Any, Tuple, Union

from .config import DEFAULT_POOL_RATIO, MAX_POOL_SIZE
from .oscontainer import OSContainer
from .oscontainer.constants import DEFAULT_CPU_COUNT
from .utils import is_aws_lambda_environment

LOGGER = logging.getLogger(__name__)


class Future(object):
    """Encapsulates the asynchronous execution of a callable."""

    def __init__(
        self, future: concurrent.futures.Future = None, async_result: AsyncResult = None
    ) -> None:
        if future is None and async_result is None:
            raise ValueError("You need to provide either future or async_result")

        if future is not None and async_result is not None:
            raise ValueError(
                "You need to provide only a future or async_result, not both"
            )

        self._future = future
        self._async_result = async_result

    def done(self) -> bool:
        """Return True if the call was successfully cancelled or finished running."""
        if self._future is not None:
            return self._future.done()
        elif self._async_result is not None:
            return self._async_result.ready()
        else:
            self._raise_inconsistent_state()

    def successful(self) -> bool:
        """Return whether the call completed without raising an exception."""
        if self._future is not None:
            try:
                return self._future.exception() is None
            except (CancelledError, TimeoutError):
                return False
        elif self._async_result is not None:
            self._async_result.wait()
            return self._async_result.successful()
        else:
            self._raise_inconsistent_state()

    def result(self, timeout: Union[int, float] = None) -> Any:
        """Return the value returned by the call. If the call has not yet completed then this method will
        wait up to timeout seconds."""
        if self._future is not None:
            return self._future.result(timeout=timeout)
        elif self._async_result is not None:
            return self._async_result.get(timeout=timeout)
        else:
            self._raise_inconsistent_state()

    def _raise_inconsistent_state(self):
        raise RuntimeError(
            "Inconsistent state, neither future nor async_result wrapped."
        )


class ThreadPoolExecutor(object):
    """An Executor subclass that uses a pool of at most max_workers threads to execute calls asynchronously."""

    def __init__(self, max_workers: int, use_concurrent_executor: bool) -> None:
        self.max_workers = max_workers
        self.use_concurrent_executor = use_concurrent_executor
        if use_concurrent_executor:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            )
        else:
            self._pool = multiprocessing.pool.ThreadPool(processes=max_workers)

    def submit(self, fn, *args, **kwargs) -> Future:
        """Schedules the callable, fn, to be executed as fn(*args **kwargs) and returns a Future
        object representing the execution of the callable."""
        if self.use_concurrent_executor:
            future = self._executor.submit(fn, *args, **kwargs)
            return Future(future=future)
        else:
            async_result = self._pool.apply_async(fn, args=args, kwds=kwargs)
            return Future(async_result=async_result)

    def close(self) -> None:
        if self.use_concurrent_executor:
            self._executor.shutdown(wait=False)
        else:
            self._pool.close()

    def join(self) -> None:
        if self.use_concurrent_executor:
            self._executor.shutdown(wait=True)
        else:
            self._pool.join()


def get_thread_pool(
    worker_cpu_ratio: int, worker_count: int = None, os_container: OSContainer = None
) -> Tuple[int, int, ThreadPoolExecutor]:
    """Allows to get initialized ThreadPoolExecutor with specified parameters.
    Returns: Tuple with calculated pool size, detected number of CPUs, and ThreadPoolExecutor instance.

    """
    if os_container is None:
        os_container = OSContainer()
    if os_container.is_containerized():
        try:
            cpu_count = os_container.active_processor_count()
        except Exception:
            LOGGER.error(
                "Failed to calculate active processors count. Fall back to default CPU count %d"
                % DEFAULT_CPU_COUNT,
                exc_info=True,
            )
            cpu_count = DEFAULT_CPU_COUNT
    else:
        try:
            cpu_count = multiprocessing.cpu_count() or 1
        except NotImplementedError:
            # os.cpu_count is not available on Python 2 and multiprocessing.cpu_count can raise NotImplementedError
            cpu_count = DEFAULT_CPU_COUNT

    if worker_count is not None:
        pool_size = worker_count
    else:
        if not isinstance(worker_cpu_ratio, int) or worker_cpu_ratio <= 0:
            LOGGER.debug("Invalid worker_cpu_ratio %r", worker_cpu_ratio)
            worker_cpu_ratio = DEFAULT_POOL_RATIO

        pool_size = min(MAX_POOL_SIZE, cpu_count * worker_cpu_ratio)

    # In AWS lambda we use ThreadPoolExecutor implementation backed by concurrent.futures.ThreadPoolExecutor
    # because multiprocessing.pool.Pool is not supported there. Meantime, for all other environments we are using
    # multiprocessing.pool.Pool because with concurrent.futures.ThreadPoolExecutor we got some mystical experiment
    # cleanup time issues in Python 3.9+.
    # For more details see: https://comet-ml.atlassian.net/browse/CM-4449
    use_concurrent_executor = is_aws_lambda_environment()

    return (
        pool_size,
        cpu_count,
        ThreadPoolExecutor(
            max_workers=pool_size, use_concurrent_executor=use_concurrent_executor
        ),
    )
