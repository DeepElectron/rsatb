#ifndef RSATB_UTILS_INTERRUPT_HPP
#define RSATB_UTILS_INTERRUPT_HPP

// Cooperative Ctrl-C for the compute loops.
//
// Why this exists: a task runs as ONE C++ call from Python, and CPython's
// SIGINT handling is deferred -- the C handler only sets a flag, and
// KeyboardInterrupt is raised at the next bytecode. During a task no bytecode
// runs for minutes to hours, so Ctrl-C appears to do nothing and the process
// can only be killed with SIGTERM, losing every result. (SIGTERM works
// because its default action is enforced by the kernel; SIGINT is the one
// Python catches and defers.)
//
// The fix is the standard one for MPI codes: a guard installs a handler that
// records the signal, and the long loops poll at their step boundaries. The
// poll is COLLECTIVE -- the flag is max-reduced over the communicator -- so
// every rank agrees on the same step, throws together, and no rank is left
// waiting in a collective that its peers abandoned. mpirun forwards SIGINT to
// every rank, but delivery is not simultaneous; the reduce is what turns
// "most ranks know" into "all ranks act".
//
// The poll costs one small Allreduce per step. At a step boundary that is
// noise against the matvecs the step just did; inside a matvec it would not
// be, which is why polls sit only at step granularity.
//
// A second Ctrl-C exits immediately (130): the user pressing twice is saying
// the graceful path is too slow.

#include <mpi.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <stdexcept>

namespace rsatb {
namespace util {

/// Thrown at the first step boundary after Ctrl-C. Mapped to Python's
/// KeyboardInterrupt by the bindings, so the CLI sees the ordinary thing.
struct InterruptedError : std::runtime_error {
    InterruptedError() : std::runtime_error("interrupted (Ctrl-C)") {}
};

namespace detail_interrupt {

inline std::atomic<int>& flag() {
    static std::atomic<int> value{0};
    return value;
}

inline void handler(int) {
    // Async-signal-safe: an atomic exchange and, on the second press,
    // _Exit -- which is on the safe list, unlike exit().
    if (flag().exchange(1) != 0) std::_Exit(130);
}

inline int& depth() {
    static int value = 0;
    return value;
}

inline void (*&saved_handler())(int) {
    static void (*value)(int) = SIG_DFL;
    return value;
}

}  // namespace detail_interrupt

/// Installs the SIGINT recorder for a compute region, restoring whatever was
/// there before (Python's own handler, under the CLI) when the outermost
/// guard leaves. Nested guards are counted, not re-installed, so a task that
/// calls another task (quantum_metric -> ac) keeps one consistent handler.
class InterruptGuard {
public:
    InterruptGuard() {
        if (detail_interrupt::depth()++ == 0) {
            detail_interrupt::saved_handler() =
                std::signal(SIGINT, &detail_interrupt::handler);
        }
    }

    ~InterruptGuard() {
        if (--detail_interrupt::depth() == 0) {
            std::signal(SIGINT, detail_interrupt::saved_handler());
            // A signal that arrived but was never polled dies with the region
            // it aimed at; carrying it into the next task would cancel work
            // the user had not seen start yet.
            detail_interrupt::flag().store(0);
        }
    }

    InterruptGuard(const InterruptGuard&) = delete;
    InterruptGuard& operator=(const InterruptGuard&) = delete;
};

/// The step-boundary poll. Collective over `comm`: every rank must call it at
/// the same point of the same iteration, which the compute loops already
/// guarantee -- their iteration counts come from collectively-agreed data.
inline void throw_if_interrupted(MPI_Comm comm) {
    int local = detail_interrupt::flag().load(std::memory_order_relaxed);
    int global = local;
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF) {
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm);
    }
    if (global != 0) throw InterruptedError();
}

}  // namespace util
}  // namespace rsatb

#endif  // RSATB_UTILS_INTERRUPT_HPP
