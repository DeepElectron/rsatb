#ifndef RSATB_UTILS_PROFILE_HPP
#define RSATB_UTILS_PROFILE_HPP

// Levelled logging and a phase profiler.
//
// Why this exists: before it, `verbose` was a single bool threaded through 34
// signatures, timing existed in exactly two files, and there was no way to ask
// where a run spent its time. Answering that meant patching printf probes into
// the source and rebuilding -- which is how the 55s Hamiltonian read and the
// 662s Newton-Schulz refinement were eventually found. Both were invisible
// from the outside.
//
// Cost. A scope costs two omp_get_wtime() calls (~20-30 ns each) plus one
// addition into a fixed slot. That is why the rule below is a rule and not a
// preference:
//
//     Instrument PHASES, never inner loops.
//
// A phase is something that happens once per setup, once per task, or once per
// iteration of an outer loop -- the Hamiltonian read, the seed, one
// Newton-Schulz step, one task. Never a matvec, never a block, never anything
// inside an OpenMP loop over subgraphs. At phase granularity the overhead is
// unmeasurable against the seconds being measured; inside a matvec it would be
// the thing being measured.
//
// Levels are cumulative: each shows everything below it.
//
//     kSilent   nothing
//     kSummary  one line per phase boundary, plus the profile table at the end
//     kProgress adds per-iteration progress (Newton-Schulz steps, task batches)
//     kDebug    adds everything the old `verbose` bool printed
//
// The old bool maps onto kDebug, so existing call sites keep their behaviour
// until they are ported.

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rsatb {
namespace util {

inline double wall_time() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return MPI_Wtime();
#endif
}

/// Resident set size in KiB, or 0 where /proc is not available.
///
/// statm rather than status: two integers to parse instead of forty lines, and
/// the sampler below reads it fifty times a second.
inline long long read_rss_kb() {
    std::ifstream statm("/proc/self/statm");
    if (!statm) return 0;
    long long total_pages = 0, resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) return 0;
    static const long long page_kb = 4;  // Linux base page; hugepages still count in 4K units here
    return resident_pages * page_kb;
}

/// Highest resident set size seen inside each phase.
///
/// Why a sampler and not a reading at each phase boundary: the allocations that
/// decide whether a run fits are TRANSIENT. A Hermitian product holds its upper
/// triangle and its completion at the same time and frees one of them before it
/// returns, so entry/exit readings show nothing while the process really did
/// touch 1.5x the result. Only something watching from outside sees that.
///
/// Cost: one thread reading two integers every 20 ms. Against phases that run
/// for seconds this is unmeasurable, and it does no MPI, so it is safe under
/// MPI_THREAD_FUNNELED. It will miss a spike shorter than the interval; phases
/// here are orders of magnitude longer.
class MemorySampler {
public:
    static MemorySampler& instance() {
        static MemorySampler sampler;
        return sampler;
    }

    /// Idempotent: the first caller starts the thread, later ones do nothing.
    void start() {
        if (running_.exchange(true)) return;
        watermark_.store(read_rss_kb());
        thread_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                raise(read_rss_kb());
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }

    bool running() const { return running_.load(); }

    void raise(long long value_kb) {
        long long current = watermark_.load(std::memory_order_relaxed);
        while (value_kb > current &&
               !watermark_.compare_exchange_weak(current, value_kb)) {
        }
        long long high = global_peak_.load(std::memory_order_relaxed);
        while (value_kb > high &&
               !global_peak_.compare_exchange_weak(high, value_kb)) {
        }
    }

    /// Start a fresh watermark and return the one it replaces, so a phase can
    /// restore its parent's on the way out.
    long long restart_watermark() { return watermark_.exchange(read_rss_kb()); }

    long long watermark_kb() const { return watermark_.load(); }

    /// Never reset, unlike the per-phase watermark. This is what a progress
    /// line reports: a run that dies inside a phase never reaches the profile
    /// table, so the only memory number it will ever emit is the one printed
    /// while it was still alive.
    long long global_peak_kb() const { return global_peak_.load(); }

private:
    MemorySampler() = default;
    ~MemorySampler() {
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
    }

    std::atomic<long long> watermark_{0};
    std::atomic<long long> global_peak_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

enum class LogLevel : int {
    kSilent = 0,
    kSummary = 1,
    kProgress = 2,
    kDebug = 3,
};

/// Process-wide verbosity and the rank allowed to print. Set once at setup.
class Log {
public:
    /// Defaults to kSummary rather than kSilent: a library that says nothing
    /// about a setup that took twenty minutes is not being quiet, it is being
    /// unhelpful. Callers that want silence ask for it.
    static LogLevel& level() {
        static LogLevel value = LogLevel::kSummary;
        return value;
    }

    /// Only this rank prints. -1 means "every rank", which is for debugging a
    /// specific rank's behaviour and is deliberately not the default.
    static int& print_rank() {
        static int value = 0;
        return value;
    }

    static int& rank() {
        static int value = 0;
        return value;
    }

    static bool enabled(LogLevel at) {
        return static_cast<int>(level()) >= static_cast<int>(at) &&
               (print_rank() < 0 || rank() == print_rank());
    }
};

/// One measured phase. Accumulates across repeats so an outer loop reports a
/// total and a count rather than a line per pass.
struct PhaseRecord {
    std::string name;
    double seconds = 0.0;
    /// Team mean, filled only by the cross-rank reduce; equals `seconds` on one
    /// rank. Reported next to the max so imbalance is visible.
    double mean_seconds = 0.0;
    long long calls = 0;
    /// Highest resident set size seen while this phase was on the stack, over
    /// its calls and -- after the cross-rank reduce -- over the team. The max
    /// and not the mean, because the rank that runs out of memory is the one
    /// that decides whether the job survives.
    long long peak_kb = 0;
};

/// Phase totals for this rank. A plain vector with a linear scan: the number of
/// distinct phases is a handful, so a hash would cost more than it saves, and a
/// scan of ten strings is nothing against a phase measured in seconds.
class Profile {
public:
    static std::vector<PhaseRecord>& records() {
        static std::vector<PhaseRecord> value;
        return value;
    }

    static bool& enabled() {
        static bool value = false;
        return value;
    }

    /// Thread-safe, because nothing stops a phase from being opened inside an
    /// OpenMP region and an unsynchronised push_back onto a shared vector is a
    /// data race, not a rounding error. The critical section is free at the
    /// granularity this is allowed to be used at -- a phase runs for seconds,
    /// so contention on a once-per-phase lock is unmeasurable. It would not be
    /// free inside a matvec, which is one more reason that is forbidden.
    static void add(const std::string& name, double seconds, long long peak_kb) {
        #pragma omp critical(rsatb_profile_add)
        {
            auto& all = records();
            bool found = false;
            for (auto& record : all) {
                if (record.name == name) {
                    record.seconds += seconds;
                    record.calls += 1;
                    // Seconds accumulate across repeats; the peak does not --
                    // two calls each touching 8 GB did not touch 16.
                    record.peak_kb = std::max(record.peak_kb, peak_kb);
                    found = true;
                    break;
                }
            }
            if (!found) {
                PhaseRecord record;
                record.name = name;
                record.seconds = seconds;
                record.calls = 1;
                record.peak_kb = peak_kb;
                all.push_back(std::move(record));
            }
        }
    }

    /// Wall clock at the last reset, so a report can date itself rather than
    /// making every caller carry a start time around.
    static double& start_time() {
        static double value = 0.0;
        return value;
    }

    static void reset() {
        records().clear();
        start_time() = wall_time();
    }

    /// Table of where the run went, sorted by cost.
    ///
    /// `total` is the wall time percentages are taken against; pass <= 0 to use
    /// the time since reset(), which is what a caller wants unless it is
    /// reporting on a sub-interval it timed itself.
    ///
    /// `comm` is reduced over when it holds more than one rank: a table from
    /// rank 0 alone describes rank 0 alone, and the interesting question in a
    /// parallel run is usually whether the ranks agree. Both the slowest rank
    /// and the mean are shown, because their gap IS the load imbalance -- a
    /// phase at 40s max and 8s mean is a straggler, not a hot spot, and the two
    /// call for opposite fixes.
    static void report(double total = 0.0, MPI_Comm comm = MPI_COMM_NULL) {
        // Gate on the level BEFORE reducing and on the rank AFTER: the level is
        // process-wide, so every rank agrees and skipping together is safe,
        // whereas returning early on rank alone would leave the printing rank
        // inside collectives no one else entered.
        if (static_cast<int>(Log::level()) < static_cast<int>(LogLevel::kSummary)) return;
        auto all = reduced(comm);
        const int ranks = rank_count(comm);
        if (!Log::enabled(LogLevel::kSummary)) return;
        if (all.empty()) return;

        if (total <= 0.0) total = wall_time() - start_time();

        double sum = 0.0;
        for (const auto& record : all) sum += record.seconds;
        // The reduce takes each phase's max over ranks, and different phases
        // can peak on different ranks -- so the column may sum past any single
        // rank's wall clock. Taking the larger of the two as the denominator
        // keeps every percentage <= 100 instead of printing a table that
        // appears to account for more time than the run took.
        const double denominator = std::max(total, sum);
        if (!(denominator > 0.0)) return;

        std::sort(all.begin(), all.end(),
                  [](const PhaseRecord& a, const PhaseRecord& b) {
                      return a.seconds > b.seconds;
                  });

        const bool parallel = ranks > 1;
        const bool have_memory = std::any_of(
            all.begin(), all.end(),
            [](const PhaseRecord& r) { return r.peak_kb > 0; });
        std::printf("\nprofile: %.2fs total, %.2fs in named phases"
                    " (self time, nested phases subtracted)%s\n",
                    total, sum, parallel ? ", slowest rank of the team" : "");
        if (have_memory) {
            // Said once rather than implied by the column header: a reader who
            // adds the column up and compares it with the node's memory will
            // otherwise conclude the run needs several times what it does.
            std::printf("         peak(GB) is the highest resident set inside each"
                        " phase%s; peaks do not sum\n",
                        parallel ? ", over the team" : "");
        }
        if (parallel) {
            std::printf("  %-34s %10s %10s %9s %7s %8s\n",
                        "phase", "max(s)", "mean(s)", "peak(GB)", "%", "calls");
        } else {
            std::printf("  %-34s %10s %9s %7s %8s\n",
                        "phase", "wall(s)", "peak(GB)", "%", "calls");
        }
        for (const auto& record : all) {
            const double share = 100.0 * record.seconds / denominator;
            const double peak_gb = record.peak_kb / 1048576.0;
            if (parallel) {
                std::printf("  %-34s %10.2f %10.2f %9.2f %6.1f%% %8lld\n", record.name.c_str(),
                            record.seconds, record.mean_seconds, peak_gb, share, record.calls);
            } else {
                std::printf("  %-34s %10.2f %9.2f %6.1f%% %8lld\n", record.name.c_str(),
                            record.seconds, peak_gb, share, record.calls);
            }
        }
        if (total > sum) {
            // Always shown. An unattributed remainder is the thing worth
            // chasing next, and a table that hides it reads as full coverage.
            if (parallel) {
                std::printf("  %-34s %10.2f %10s %9s %6.1f%% %8s\n", "(unaccounted)",
                            total - sum, "-", "-", 100.0 * (total - sum) / denominator, "-");
            } else {
                std::printf("  %-34s %10.2f %9s %6.1f%% %8s\n", "(unaccounted)",
                            total - sum, "-", 100.0 * (total - sum) / denominator, "-");
            }
        }
        std::fflush(stdout);
    }

private:
    static int rank_count(MPI_Comm comm) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized || comm == MPI_COMM_NULL) return 1;
        int size = 1;
        MPI_Comm_size(comm, &size);
        return size;
    }

    /// This rank's records, or the team's when `comm` spans more than one.
    ///
    /// Ranks need not have run the same phases -- a rank owning no k-points
    /// never enters the preconditioner -- so the names are unioned rather than
    /// assumed to match. Getting that wrong would deadlock rather than mislead,
    /// since the collectives below are sized by the union.
    static std::vector<PhaseRecord> reduced(MPI_Comm comm) {
        auto local = records();
        for (auto& record : local) record.mean_seconds = record.seconds;

        const int ranks = rank_count(comm);
        if (ranks <= 1) return local;

        const std::vector<std::string> names = union_of_names(local, comm);
        const int n = static_cast<int>(names.size());
        if (n == 0) return {};

        std::vector<double> seconds(n, 0.0);
        std::vector<long long> calls(n, 0);
        std::vector<long long> peaks(n, 0);
        for (int i = 0; i < n; ++i) {
            for (const auto& record : local) {
                if (record.name == names[i]) {
                    seconds[i] = record.seconds;
                    calls[i] = record.calls;
                    peaks[i] = record.peak_kb;
                    break;
                }
            }
        }

        std::vector<double> max_seconds(n, 0.0), sum_seconds(n, 0.0);
        std::vector<long long> max_calls(n, 0), max_peaks(n, 0);
        MPI_Allreduce(seconds.data(), max_seconds.data(), n, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(seconds.data(), sum_seconds.data(), n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(calls.data(), max_calls.data(), n, MPI_LONG_LONG, MPI_MAX, comm);
        MPI_Allreduce(peaks.data(), max_peaks.data(), n, MPI_LONG_LONG, MPI_MAX, comm);

        std::vector<PhaseRecord> merged;
        merged.reserve(n);
        for (int i = 0; i < n; ++i) {
            PhaseRecord record;
            record.name = names[i];
            record.seconds = max_seconds[i];
            record.mean_seconds = sum_seconds[i] / ranks;
            record.calls = max_calls[i];
            record.peak_kb = max_peaks[i];
            merged.push_back(std::move(record));
        }
        return merged;
    }

    /// Every phase name any rank recorded, in a single agreed order.
    ///
    /// Names are packed newline-separated and gathered as bytes: there are a
    /// handful of them, once per run, so the simple encoding costs nothing and
    /// avoids a second round of length exchanges.
    static std::vector<std::string> union_of_names(const std::vector<PhaseRecord>& local,
                                                   MPI_Comm comm) {
        std::string packed;
        for (const auto& record : local) {
            packed += record.name;
            packed += '\n';
        }

        int ranks = 1;
        MPI_Comm_size(comm, &ranks);
        int length = static_cast<int>(packed.size());
        std::vector<int> lengths(ranks, 0);
        MPI_Allgather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, comm);

        std::vector<int> offsets(ranks, 0);
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            offsets[r] = total;
            total += lengths[r];
        }
        std::string gathered(static_cast<size_t>(total), '\0');
        MPI_Allgatherv(packed.data(), length, MPI_CHAR, &gathered[0],
                       lengths.data(), offsets.data(), MPI_CHAR, comm);

        std::vector<std::string> names;
        size_t begin = 0;
        while (begin < gathered.size()) {
            const size_t end = gathered.find('\n', begin);
            if (end == std::string::npos) break;
            std::string name = gathered.substr(begin, end - begin);
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                names.push_back(std::move(name));
            }
            begin = end + 1;
        }
        return names;
    }
};

/// A fault -- a breakdown, a divergence, a series that did not converge -- is
/// not progress and is not subject to the level. Silencing one would hide the
/// reason a result is wrong, so these print at every level, and on stderr so
/// they can be filtered and redirected apart from the run's output. Still
/// rank-gated: 96 ranks reporting the same breakdown is not 96 times as useful.
inline bool fault() {
    return Log::print_rank() < 0 || Log::rank() == Log::print_rank();
}

/// True on the first index and then every `stride`, for turning a print that
/// fires once per iteration into one that fires a bounded number of times. A
/// 1000-step task emitting 1000 lines is not progress reporting, it is noise
/// that buries the lines that matter.
inline bool every_nth(long long index, long long stride) {
    return stride <= 1 || index == 0 || (index % stride) == 0;
}

/// Times its scope into the named phase. Does nothing when profiling is off,
/// so leaving these in place costs a predictable branch.
///
/// Records SELF time, not elapsed time: whatever nested scopes consumed is
/// subtracted out. Without that a phase containing another phase counts its
/// child twice and the table sums past 100%, which is exactly what the
/// inverse-square-root retry does -- it re-enters the whole seed-and-refine
/// sequence from inside the refinement it is retrying. Self time makes the
/// column additive, so a phase's number is work that phase actually did.
class ScopedPhase {
public:
    explicit ScopedPhase(std::string name)
        : name_(std::move(name)),
          start_(Profile::enabled() ? wall_time() : 0.0),
          saved_child_(child_seconds()) {
        child_seconds() = 0.0;
        if (Profile::enabled()) {
            // Fresh watermark for this phase; the parent's is restored below.
            saved_watermark_ = MemorySampler::instance().restart_watermark();
            if (per_call_detail()) entry_kb_ = read_rss_kb();
        }
    }

    ~ScopedPhase() {
        if (!Profile::enabled()) return;
        const double elapsed = wall_time() - start_;
        // Memory is reported as the peak while this phase was on the stack --
        // NOT self time's analogue. Peaks do not decompose: a parent's peak IS
        // its tallest child's, and subtracting children would report a parent
        // that never happened.
        const long long peak = MemorySampler::instance().watermark_kb();
        Profile::add(name_, elapsed - child_seconds(), peak);

        // The phase table keeps the MAX over a phase's calls, which is the
        // right summary and the wrong diagnostic: in a loop whose matrices grow
        // and collapse across steps it cannot say WHICH call peaked, so a peak
        // and the state that produced it can never be lined up. This prints one
        // line per call -- entry, peak, exit -- so the transient (peak - entry)
        // is attributable to a single step. Off unless RSATB_PHASE_CALLS is set,
        // because at that granularity it is a trace, not a report.
        if (per_call_detail() && Log::enabled(LogLevel::kProgress)) {
            const long long exit_kb = read_rss_kb();
            std::printf("phase-call %-32s entry %8.2f  peak %8.2f  exit %8.2f GB"
                        "  transient %8.2f GB  self %9.2fs\n",
                        name_.c_str(), entry_kb_ / 1048576.0, peak / 1048576.0,
                        exit_kb / 1048576.0, (peak - entry_kb_) / 1048576.0,
                        elapsed - child_seconds());
            std::fflush(stdout);
        }

        // Hand our whole span to the parent so it can subtract us in turn.
        child_seconds() = saved_child_ + elapsed;
        // ... and our peak, so the parent's covers ours.
        MemorySampler::instance().raise(std::max(saved_watermark_, peak));
    }

    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;

private:
    /// Time consumed by scopes nested inside the one currently open. Per
    /// thread, so that a phase opened inside an OpenMP region subtracts its
    /// own children rather than whichever thread's happened to land last.
    static double& child_seconds() {
        static thread_local double value = 0.0;
        return value;
    }

    static bool per_call_detail() {
        static const bool on = std::getenv("RSATB_PHASE_CALLS") != nullptr;
        return on;
    }

    std::string name_;
    double start_;
    double saved_child_;
    long long saved_watermark_ = 0;
    long long entry_kb_ = 0;
};

}  // namespace util
}  // namespace rsatb

// Two levels of indirection are required: ## suppresses expansion of its
// operands, so pasting __LINE__ directly yields the literal token
// `rsatb_phase___LINE__` and every site in a scope collides.
#define RSATB_PHASE_PASTE_(a, b) a##b
#define RSATB_PHASE_NAME_(a, b) RSATB_PHASE_PASTE_(a, b)
#define RSATB_PHASE(name) \
    ::rsatb::util::ScopedPhase RSATB_PHASE_NAME_(rsatb_phase_, __LINE__)(name)

#endif  // RSATB_UTILS_PROFILE_HPP
