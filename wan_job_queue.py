"""
Persistent Job Queue System for Wan2.2 Video Generation

This module provides a file-based job queue that operates independently of Gradio.
Jobs persist across server restarts and continue processing even when browsers disconnect.

Adapted from Kandinsky5 job_queue.py for Wan2.2 video generation pipeline.

Cross-platform compatible (Windows and Linux).
"""

import json
import os
import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from enum import Enum
from datetime import datetime
from contextlib import contextmanager


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a video generation job."""
    id: str
    created_at: float
    status: str = JobStatus.PENDING.value

    # Generation parameters
    command: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Progress tracking
    progress: float = 0.0
    progress_text: str = ""
    current_step: int = 0
    total_steps: int = 0

    # Output
    output_filename: str = ""
    preview_path: str = ""

    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    elapsed_time: float = 0.0

    # Error handling
    error_message: str = ""
    return_code: Optional[int] = None

    # Process tracking
    process_id: Optional[int] = None

    # Batch tracking
    batch_id: Optional[str] = None
    batch_index: int = 0
    batch_total: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create job from dictionary."""
        return cls(**data)


class FileLock:
    """
    Cross-platform file lock implementation.
    Uses lock files rather than OS-specific file locking.
    """

    def __init__(self, lock_file: str, timeout: float = 10.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self._lock_acquired = False

    def acquire(self) -> bool:
        """Try to acquire the lock."""
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                # Try to create lock file exclusively
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._lock_acquired = True
                return True
            except FileExistsError:
                # Lock file exists - check if the process holding it is still alive
                try:
                    with open(self.lock_file, 'r') as f:
                        pid = int(f.read().strip())
                    # Check if process is still running (works on both Windows and Linux)
                    if not self._is_process_running(pid):
                        # Stale lock - remove it
                        os.remove(self.lock_file)
                        continue
                except (ValueError, FileNotFoundError, PermissionError):
                    # Lock file corrupted or disappeared - try again
                    try:
                        os.remove(self.lock_file)
                    except:
                        pass
                    continue

                # Process is still running, wait a bit
                time.sleep(0.05)
            except Exception as e:
                time.sleep(0.05)

        return False

    def release(self) -> None:
        """Release the lock."""
        if self._lock_acquired:
            try:
                os.remove(self.lock_file)
            except:
                pass
            self._lock_acquired = False

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process is running."""
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:  # Unix
                os.kill(pid, 0)
                return True
        except (OSError, ProcessLookupError):
            return False

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.lock_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class JobQueue:
    """
    Persistent job queue with file-based storage.

    Uses file locking to ensure safe concurrent access from multiple processes
    (Gradio frontend + background worker).
    """

    def __init__(self, queue_file: str = "wan_job_queue.json", lock_timeout: float = 10.0):
        self.queue_file = queue_file
        self.lock_file = queue_file + ".lock"
        self.lock_timeout = lock_timeout
        self._thread_lock = threading.Lock()

        # Initialize empty queue file if it doesn't exist
        if not os.path.exists(self.queue_file):
            self._save_jobs({})

    @contextmanager
    def _file_lock(self):
        """Context manager for file locking."""
        lock = FileLock(self.lock_file, self.lock_timeout)
        try:
            lock.acquire()
            yield
        finally:
            lock.release()

    def _load_jobs(self) -> Dict[str, Dict]:
        """Load all jobs from the queue file."""
        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return {}
                return json.loads(content)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Corrupted queue file, returning empty queue")
            return {}

    def _save_jobs(self, jobs: Dict[str, Dict]) -> None:
        """Save all jobs to the queue file."""
        # Write to temp file first, then rename (atomic on most systems)
        temp_file = self.queue_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, indent=2)

        # On Windows, need to remove target first
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)
        os.rename(temp_file, self.queue_file)

    def add_job(self,
                command: List[str],
                parameters: Dict[str, Any],
                output_filename: str,
                batch_id: Optional[str] = None,
                batch_index: int = 0,
                batch_total: int = 1) -> Job:
        """
        Add a new job to the queue.

        Args:
            command: The full command to execute (e.g., ['python', 'wan2_generate_video.py', ...])
            parameters: Dictionary of generation parameters for display/metadata
            output_filename: Expected output video path
            batch_id: Optional batch identifier for grouped jobs
            batch_index: Position in batch (0-indexed)
            batch_total: Total jobs in batch

        Returns:
            The created Job object
        """
        with self._thread_lock:
            with self._file_lock():
                job = Job(
                    id=str(uuid.uuid4())[:8],
                    created_at=time.time(),
                    command=command,
                    parameters=parameters,
                    output_filename=output_filename,
                    batch_id=batch_id or str(uuid.uuid4())[:8],
                    batch_index=batch_index,
                    batch_total=batch_total,
                )

                jobs = self._load_jobs()
                jobs[job.id] = job.to_dict()
                self._save_jobs(jobs)

                return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by its ID."""
        with self._file_lock():
            jobs = self._load_jobs()
            if job_id in jobs:
                return Job.from_dict(jobs[job_id])
            return None

    def update_job(self, job_id: str, **updates) -> Optional[Job]:
        """
        Update job fields.

        Args:
            job_id: The job ID to update
            **updates: Field names and values to update

        Returns:
            Updated Job object or None if not found
        """
        with self._thread_lock:
            with self._file_lock():
                jobs = self._load_jobs()
                if job_id not in jobs:
                    return None

                jobs[job_id].update(updates)
                self._save_jobs(jobs)

                return Job.from_dict(jobs[job_id])

    def get_next_pending(self) -> Optional[Job]:
        """
        Get the next pending job in the queue (FIFO order).

        Returns:
            The oldest pending job or None if queue is empty
        """
        with self._file_lock():
            jobs = self._load_jobs()
            pending = [
                Job.from_dict(j) for j in jobs.values()
                if j['status'] == JobStatus.PENDING.value
            ]

            if not pending:
                return None

            # Sort by creation time and return oldest
            pending.sort(key=lambda x: x.created_at)
            return pending[0]

    def get_running_jobs(self) -> List[Job]:
        """Get all currently running jobs."""
        with self._file_lock():
            jobs = self._load_jobs()
            return [
                Job.from_dict(j) for j in jobs.values()
                if j['status'] == JobStatus.RUNNING.value
            ]

    def get_all_jobs(self, limit: int = 100) -> List[Job]:
        """
        Get all jobs, most recent first.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of Job objects sorted by creation time (newest first)
        """
        with self._file_lock():
            jobs = self._load_jobs()
            all_jobs = [Job.from_dict(j) for j in jobs.values()]
            all_jobs.sort(key=lambda x: x.created_at, reverse=True)
            return all_jobs[:limit]

    def get_jobs_by_status(self, status: JobStatus, limit: int = 50) -> List[Job]:
        """Get jobs with a specific status."""
        with self._file_lock():
            jobs = self._load_jobs()
            filtered = [
                Job.from_dict(j) for j in jobs.values()
                if j['status'] == status.value
            ]
            filtered.sort(key=lambda x: x.created_at, reverse=True)
            return filtered[:limit]

    def get_batch_jobs(self, batch_id: str) -> List[Job]:
        """Get all jobs in a batch."""
        with self._file_lock():
            jobs = self._load_jobs()
            batch_jobs = [
                Job.from_dict(j) for j in jobs.values()
                if j.get('batch_id') == batch_id
            ]
            batch_jobs.sort(key=lambda x: x.batch_index)
            return batch_jobs

    def cancel_job(self, job_id: str) -> Optional[Job]:
        """
        Cancel a pending or running job.

        For running jobs, this just marks them as cancelled.
        The worker process is responsible for checking this status and terminating.
        """
        with self._thread_lock:
            with self._file_lock():
                jobs = self._load_jobs()
                if job_id not in jobs:
                    return None

                job = jobs[job_id]
                if job['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    return Job.from_dict(job)  # Already finished, can't cancel

                job['status'] = JobStatus.CANCELLED.value
                job['completed_at'] = time.time()
                self._save_jobs(jobs)

                return Job.from_dict(job)

    def cancel_batch(self, batch_id: str) -> List[Job]:
        """Cancel all jobs in a batch."""
        batch_jobs = self.get_batch_jobs(batch_id)
        cancelled = []
        for job in batch_jobs:
            result = self.cancel_job(job.id)
            if result:
                cancelled.append(result)
        return cancelled

    def mark_running(self, job_id: str, process_id: int) -> Optional[Job]:
        """Mark a job as running with its process ID."""
        return self.update_job(
            job_id,
            status=JobStatus.RUNNING.value,
            started_at=time.time(),
            process_id=process_id
        )

    def mark_completed(self, job_id: str, return_code: int = 0) -> Optional[Job]:
        """Mark a job as completed."""
        job = self.get_job(job_id)
        if not job:
            return None

        elapsed = time.time() - (job.started_at or job.created_at)
        return self.update_job(
            job_id,
            status=JobStatus.COMPLETED.value,
            completed_at=time.time(),
            elapsed_time=elapsed,
            return_code=return_code,
            progress=100.0
        )

    def mark_failed(self, job_id: str, error_message: str, return_code: int = -1) -> Optional[Job]:
        """Mark a job as failed with an error message."""
        job = self.get_job(job_id)
        if not job:
            return None

        elapsed = time.time() - (job.started_at or job.created_at)
        return self.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            completed_at=time.time(),
            elapsed_time=elapsed,
            error_message=error_message,
            return_code=return_code
        )

    def update_progress(self, job_id: str, progress: float, progress_text: str = "",
                       current_step: int = 0, total_steps: int = 0,
                       preview_path: str = "") -> Optional[Job]:
        """Update job progress."""
        updates = {
            'progress': progress,
            'progress_text': progress_text,
            'current_step': current_step,
            'total_steps': total_steps,
        }
        if preview_path:
            updates['preview_path'] = preview_path

        return self.update_job(job_id, **updates)

    def cleanup_old_jobs(self, max_age_hours: float = 24.0) -> int:
        """
        Remove completed/failed/cancelled jobs older than max_age_hours.

        Returns:
            Number of jobs removed
        """
        with self._thread_lock:
            with self._file_lock():
                jobs = self._load_jobs()
                cutoff = time.time() - (max_age_hours * 3600)

                to_remove = []
                for job_id, job in jobs.items():
                    if job['status'] in [JobStatus.COMPLETED.value,
                                         JobStatus.FAILED.value,
                                         JobStatus.CANCELLED.value]:
                        completed_at = job.get('completed_at') or job.get('created_at', 0)
                        if completed_at < cutoff:
                            to_remove.append(job_id)

                for job_id in to_remove:
                    del jobs[job_id]

                if to_remove:
                    self._save_jobs(jobs)

                return len(to_remove)

    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about the queue."""
        with self._file_lock():
            jobs = self._load_jobs()
            stats = {
                'total': len(jobs),
                'pending': 0,
                'running': 0,
                'completed': 0,
                'failed': 0,
                'cancelled': 0,
            }

            for job in jobs.values():
                status = job.get('status', 'unknown')
                if status in stats:
                    stats[status] += 1

            return stats

    def clear_all(self) -> int:
        """
        Clear all jobs from the queue.
        WARNING: This cannot be undone!

        Returns:
            Number of jobs cleared
        """
        with self._thread_lock:
            with self._file_lock():
                jobs = self._load_jobs()
                count = len(jobs)
                self._save_jobs({})
                return count


# Global queue instance for easy access
_queue_instance = None
_queue_port = None

def set_queue_port(port: int):
    """Set the port to use for queue file naming. Must be called before get_queue()."""
    global _queue_port
    _queue_port = port

def get_queue(queue_file: str = None) -> JobQueue:
    """
    Get the global queue instance.

    Uses port-based queue files for multi-instance separation.
    Falls back to GPU-specific files if port not set.

    Examples:
        Port 7860 -> wan_job_queue_7860.json
        Port 7861 -> wan_job_queue_7861.json
        CUDA_VISIBLE_DEVICES=0 (no port) -> wan_job_queue_gpu0.json
        Not set -> wan_job_queue.json
    """
    global _queue_instance, _queue_port
    if _queue_instance is None:
        if queue_file is None:
            if _queue_port is not None:
                # Use port-based queue file
                queue_file = f"wan_job_queue_{_queue_port}.json"
                print(f"[wan_job_queue] Using port-specific queue: {queue_file}")
            else:
                # Fallback: auto-detect GPU-specific queue file
                cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_devices:
                    gpu_suffix = cuda_devices.replace(",", "_")
                    queue_file = f"wan_job_queue_gpu{gpu_suffix}.json"
                    print(f"[wan_job_queue] Using GPU-specific queue: {queue_file}")
                else:
                    queue_file = "wan_job_queue.json"
        _queue_instance = JobQueue(queue_file)
    return _queue_instance


def format_job_for_display(job: Job) -> str:
    """Format a job for display in the UI."""
    status_emoji = {
        JobStatus.PENDING.value: "PENDING",
        JobStatus.RUNNING.value: "RUNNING",
        JobStatus.COMPLETED.value: "DONE",
        JobStatus.FAILED.value: "FAILED",
        JobStatus.CANCELLED.value: "CANCELLED",
    }

    emoji = status_emoji.get(job.status, "?")

    # Format time
    created = datetime.fromtimestamp(job.created_at).strftime("%H:%M:%S")

    # Get short prompt
    prompt = job.parameters.get('prompt', 'No prompt')[:50]
    if len(job.parameters.get('prompt', '')) > 50:
        prompt += "..."

    # Progress info
    if job.status == JobStatus.RUNNING.value:
        progress = f" ({job.progress:.0f}%)"
    elif job.status == JobStatus.COMPLETED.value:
        progress = f" ({job.elapsed_time:.1f}s)"
    else:
        progress = ""

    return f"[{emoji}] [{job.id}] {created} - {prompt}{progress}"
