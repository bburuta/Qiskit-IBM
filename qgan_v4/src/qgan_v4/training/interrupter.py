import signal


class Interrupter:
    """Handle Ctrl+C once gracefully and twice immediately."""

    def __init__(self):
        self.kill_now = False
        self.interrupt_count = 0
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        self.interrupt_count += 1

        if self.interrupt_count == 1:
            self.kill_now = True
            print("\nInterrupter: will stop after the current epoch. Press Ctrl+C again to force quit.")
            return

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
