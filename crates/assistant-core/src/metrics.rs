use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};

#[derive(Default)]
struct Counters {
    audio_queue_depth: AtomicU64,
    audio_queue_max_depth: AtomicU64,
    audio_frames_dropped: AtomicU64,
    audio_reconnects: AtomicU64,
    stt_queue_depth: AtomicU64,
    stt_queue_max_depth: AtomicU64,
    stt_restarts: AtomicU64,
    stt_faults: AtomicU64,
    recoverable_errors: AtomicU64,
    commands_completed: AtomicU64,
    kws_last_us: AtomicU64,
    kws_max_us: AtomicU64,
    stt_last_ms: AtomicU64,
    stt_max_ms: AtomicU64,
    handler_last_ms: AtomicU64,
    handler_max_ms: AtomicU64,
}

/// Shared lock-free counters for the real-time and async parts of the core.
#[derive(Clone, Default)]
pub struct CoreMetrics(Arc<Counters>);

/// Serializable point-in-time metrics exposed through IPC.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetricsSnapshot {
    /// Frames currently waiting for the audio worker.
    pub audio_queue_depth: u64,
    /// Highest observed audio queue depth.
    pub audio_queue_max_depth: u64,
    /// Frames rejected by full callback queue.
    pub audio_frames_dropped: u64,
    /// Audio device reopen attempts.
    pub audio_reconnects: u64,
    /// Requests currently waiting for STT.
    pub stt_queue_depth: u64,
    /// Highest observed STT queue depth.
    pub stt_queue_max_depth: u64,
    /// Native STT workers recreated by the supervisor.
    pub stt_restarts: u64,
    /// STT restart budgets exhausted.
    pub stt_faults: u64,
    /// Recoverable component failures.
    pub recoverable_errors: u64,
    /// Successfully completed typed commands.
    pub commands_completed: u64,
    /// Most recent wake-word frame latency in microseconds.
    pub kws_last_us: u64,
    /// Highest wake-word frame latency in microseconds.
    pub kws_max_us: u64,
    /// Most recent STT request latency in milliseconds.
    pub stt_last_ms: u64,
    /// Highest STT request latency in milliseconds.
    pub stt_max_ms: u64,
    /// Most recent handler latency in milliseconds.
    pub handler_last_ms: u64,
    /// Highest handler latency in milliseconds.
    pub handler_max_ms: u64,
    /// Current process working set; zero when unsupported.
    #[serde(default)]
    pub process_working_set_bytes: u64,
    /// Accumulated process user plus kernel CPU time; zero when unsupported.
    #[serde(default)]
    pub process_cpu_time_ms: u64,
}

impl CoreMetrics {
    /// Records a successful audio enqueue.
    pub fn audio_queued(&self) {
        let depth = self.0.audio_queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        self.0
            .audio_queue_max_depth
            .fetch_max(depth, Ordering::Relaxed);
    }

    /// Records removal or rollback of one queued audio block.
    pub fn audio_dequeued(&self) {
        self.0.audio_queue_depth.fetch_sub(1, Ordering::Relaxed);
    }

    /// Resets current queue depth when an input stream closes.
    pub fn reset_audio_queue(&self) {
        self.0.audio_queue_depth.store(0, Ordering::Relaxed);
    }

    /// Counts one block dropped by callback backpressure.
    pub fn audio_dropped(&self) {
        self.0.audio_frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one device reopen attempt.
    pub fn audio_reconnected(&self) {
        self.0.audio_reconnects.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one STT job entering the bounded queue.
    pub fn stt_queued(&self) {
        let depth = self.0.stt_queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        self.0
            .stt_queue_max_depth
            .fetch_max(depth, Ordering::Relaxed);
    }

    /// Records one STT job leaving the bounded queue.
    pub fn stt_dequeued(&self) {
        self.0.stt_queue_depth.fetch_sub(1, Ordering::Relaxed);
    }

    /// Resets current STT queue depth when a failed worker generation is abandoned.
    pub fn reset_stt_queue(&self) {
        self.0.stt_queue_depth.store(0, Ordering::Relaxed);
    }

    /// Counts one supervised STT worker replacement.
    pub fn stt_restarted(&self) {
        self.0.stt_restarts.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one exhausted STT restart budget.
    pub fn stt_faulted(&self) {
        self.0.stt_faults.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one component recovery.
    pub fn recoverable_error(&self) {
        self.0.recoverable_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one successfully completed command.
    pub fn command_completed(&self) {
        self.0.commands_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records latest and maximum KWS frame latency.
    pub fn observe_kws(&self, micros: u64) {
        observe(&self.0.kws_last_us, &self.0.kws_max_us, micros);
    }

    /// Records latest and maximum STT latency.
    pub fn observe_stt(&self, millis: u64) {
        observe(&self.0.stt_last_ms, &self.0.stt_max_ms, millis);
    }

    /// Records latest and maximum command-handler latency.
    pub fn observe_handler(&self, millis: u64) {
        observe(&self.0.handler_last_ms, &self.0.handler_max_ms, millis);
    }

    /// Returns a serializable point-in-time snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let (process_working_set_bytes, process_cpu_time_ms) = process_usage();
        MetricsSnapshot {
            audio_queue_depth: load(&self.0.audio_queue_depth),
            audio_queue_max_depth: load(&self.0.audio_queue_max_depth),
            audio_frames_dropped: load(&self.0.audio_frames_dropped),
            audio_reconnects: load(&self.0.audio_reconnects),
            stt_queue_depth: load(&self.0.stt_queue_depth),
            stt_queue_max_depth: load(&self.0.stt_queue_max_depth),
            stt_restarts: load(&self.0.stt_restarts),
            stt_faults: load(&self.0.stt_faults),
            recoverable_errors: load(&self.0.recoverable_errors),
            commands_completed: load(&self.0.commands_completed),
            kws_last_us: load(&self.0.kws_last_us),
            kws_max_us: load(&self.0.kws_max_us),
            stt_last_ms: load(&self.0.stt_last_ms),
            stt_max_ms: load(&self.0.stt_max_ms),
            handler_last_ms: load(&self.0.handler_last_ms),
            handler_max_ms: load(&self.0.handler_max_ms),
            process_working_set_bytes,
            process_cpu_time_ms,
        }
    }
}

#[cfg(windows)]
fn process_usage() -> (u64, u64) {
    use windows_sys::Win32::{
        Foundation::FILETIME,
        System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::{GetCurrentProcess, GetProcessTimes},
        },
    };

    let process = unsafe { GetCurrentProcess() };
    let mut memory = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    let (mut creation, mut exit, mut kernel, mut user) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    // SAFETY: pointers reference correctly sized writable structures for the current process.
    let memory_ok = unsafe {
        GetProcessMemoryInfo(
            process,
            &mut memory,
            size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        )
    } != 0;
    // SAFETY: pointers reference writable FILETIME values for the current process.
    let times_ok =
        unsafe { GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user) } != 0;
    let filetime = |value: FILETIME| {
        ((u64::from(value.dwHighDateTime) << 32) | u64::from(value.dwLowDateTime)) / 10_000
    };
    (
        if memory_ok {
            memory.WorkingSetSize as u64
        } else {
            0
        },
        if times_ok {
            filetime(kernel) + filetime(user)
        } else {
            0
        },
    )
}

#[cfg(not(windows))]
fn process_usage() -> (u64, u64) {
    (0, 0)
}

fn observe(last: &AtomicU64, max: &AtomicU64, value: u64) {
    last.store(value, Ordering::Relaxed);
    max.fetch_max(value, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_track_queue_and_maximum() {
        let metrics = CoreMetrics::default();
        metrics.audio_queued();
        metrics.audio_queued();
        metrics.audio_dequeued();
        assert_eq!(metrics.snapshot().audio_queue_depth, 1);
        assert_eq!(metrics.snapshot().audio_queue_max_depth, 2);
    }
}
