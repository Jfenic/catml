import sys
import threading

from typing import Any

from automl.application.bootstrap import build_application
from automl.application.services.job_executor import JobExecutor, JobStopped
from automl.domain.jobs.job import JobStatus
from automl.infrastructure.database.sqlite_jobs import SQLiteJobRepository


class JobWorker:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        workspace, _, _ = build_application(workspace_dir)
        self.repository = SQLiteJobRepository(workspace.repository.db_path)
        self.lock_path = workspace.root_dir / "worker.lock"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease: Any = None
        self._executing = False

    def start(self, background: bool = True):
        if self._lease is not None:
            raise RuntimeError("Worker already started")
        lease = self.lock_path.open("a+")
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            except (OSError, IOError):
                lease.close()
                raise RuntimeError("Another worker owns this workspace") from None
        else:
            try:
                import fcntl
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                raise RuntimeError("Another worker owns this workspace") from None
            except ImportError:
                pass
        self._lease = lease
        self._stop.clear()
        self._thread = None
        try:
            self.repository.recover()
        except Exception:
            self._lease.close()
            self._lease = None
            raise
        if background:
            self._thread = threading.Thread(target=self._loop, name="catml-job-worker", daemon=True)
            self._thread.start()
        return self

    def run_once(self) -> bool:
        if self._lease is None:
            raise RuntimeError("Start the worker before claiming jobs")
        if self._stop.is_set():
            return False
        job = self.repository.claim()
        if job is None:
            return False
        self._executing = True
        try:
            workspace, commands, queries = build_application(self.workspace_dir)
            result = JobExecutor(workspace, commands, queries, self.repository, self._stop.is_set).execute(job)
            self.repository.finish(job.id, JobStatus.COMPLETED, result=result,
                                   message="Completed", error=None)
        except JobStopped as exc:
            self.repository.finish(job.id, exc.status, message=str(exc))
        except Exception as exc:
            self.repository.finish(job.id, JobStatus.FAILED, error=str(exc), message=str(exc))
        finally:
            self._executing = False
            if self._stop.is_set() and self._thread is None and self._lease is not None:
                self._lease.close()
                self._lease = None
        return True

    def _loop(self):
        try:
            while not self._stop.is_set():
                if not self.run_once():
                    self._stop.wait(0.25)
        finally:
            self._lease.close()
            self._lease = None

    def close(self, timeout: float | None = None) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        if self._executing:
            return False
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        return True
