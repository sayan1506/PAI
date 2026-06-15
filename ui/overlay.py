"""
Conversation Overlay

A small always-on-top tkinter window in the bottom-right corner that shows
the user's last heard phrase and PAI's last response in real time.

The overlay runs tkinter's mainloop in a daemon thread. The main thread
communicates via queue.Queue, and the overlay polls every 100ms using
root.after(). This is the canonical cross-thread tkinter pattern.

Usage:
    from ui.overlay import ConversationOverlay

    overlay = ConversationOverlay()
    overlay.start()
    overlay.update("Hello PAI", "Hi! How can I help?")
    overlay.close()
"""

import queue
import threading
from typing import Optional

import config
from core.logger import logger


class ConversationOverlay:
    """Thread-safe, always-on-top conversation overlay built on tkinter.

    Renders the most recent user phrase and PAI response in a borderless
    window in the bottom-right of the screen. The tkinter event loop runs in
    a daemon thread; callers push updates from any thread via a
    :class:`queue.Queue`, and the overlay drains the queue on a 100ms timer.

    The overlay degrades gracefully: if disabled by configuration or if no
    display/tkinter backend is available, every method becomes a no-op.

    Attributes:
        _queue: Thread-safe channel carrying ``(user_text, pai_text)`` tuples,
            or a ``None`` sentinel signalling shutdown.
        _thread: The daemon thread running tkinter's mainloop, or None.
        _root: The tkinter root window, or None until the thread starts.
        _enabled: Whether the overlay is active; cleared if startup fails.
        _running: Whether the overlay thread is currently running.
    """

    def __init__(self) -> None:
        """Initialise overlay state without creating any window.

        Reads ``config.OVERLAY_ENABLED`` to decide whether the overlay is
        active. No tkinter resources are allocated until :meth:`start` runs.
        """
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._root = None
        self._enabled: bool = config.OVERLAY_ENABLED
        self._running: bool = False

    def start(self) -> None:
        """Start the overlay window in a daemon thread.

        Probes for a usable tkinter display before spawning the thread so a
        headless environment fails gracefully rather than crashing.

        Side Effects:
            Spawns a daemon thread and sets ``_running``. No-op if
            ``OVERLAY_ENABLED`` is false; logs a warning and disables the
            overlay if no display/tkinter backend is available.
        """
        if not self._enabled:
            logger.debug("Overlay disabled via OVERLAY_ENABLED=false")
            return

        # Test if tkinter/display is available before spawning thread
        try:
            import tkinter as tk

            test_root = tk.Tk()
            test_root.destroy()
        except Exception as e:
            logger.warning(
                f"Overlay unavailable (no display or tkinter error): {e}"
            )
            self._enabled = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def update(self, user_text: str, pai_text: str) -> None:
        """Queue a new conversation pair for display.

        Thread-safe; may be called from any thread. The overlay thread picks
        the update up on its next poll.

        Args:
            user_text: The latest phrase heard from the user.
            pai_text: PAI's latest response.

        Side Effects:
            Enqueues the pair. No-op if the overlay is disabled or not running.
        """
        if not self._enabled or not self._running:
            return
        self._queue.put((user_text, pai_text))

    def close(self) -> None:
        """Signal the overlay thread to tear down its window.

        Enqueues a sentinel that causes the daemon thread to destroy the
        root window and exit its mainloop.

        Side Effects:
            Clears ``_running``. No-op if the overlay is disabled or not
            running.
        """
        if not self._enabled or not self._running:
            return
        self._queue.put(None)  # Sentinel to signal shutdown
        self._running = False

    def _run(self) -> None:
        """Daemon-thread entry point: build the window and run the mainloop.

        Imports tkinter lazily, constructs the borderless always-on-top
        window positioned in the bottom-right corner, starts the queue poll
        loop, then blocks in ``mainloop`` until the window is destroyed.

        Side Effects:
            Creates tkinter widgets and blocks the calling thread. Disables
            the overlay and returns early if tkinter is missing or the window
            cannot be created.
        """
        try:
            import tkinter as tk
        except ImportError:
            logger.warning("tkinter not available; overlay disabled")
            self._enabled = False
            return

        try:
            root = tk.Tk()
        except Exception as e:
            logger.warning(f"Cannot create overlay window: {e}")
            self._enabled = False
            return

        self._root = root

        # Window configuration: borderless, always-on-top, semi-transparent
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.92)
        root.configure(bg="#1e1e2e")

        # Window size
        width = 420
        height = 120

        # Position: bottom-right with taskbar allowance (~50px)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = screen_width - width - 20
        y = screen_height - height - 70  # 50px taskbar + 20px margin
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Content frame
        frame = tk.Frame(root, bg="#1e1e2e", padx=12, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # "You:" label
        self._you_label = tk.Label(
            frame,
            text="You: ...",
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            anchor="w",
            wraplength=390,
            justify="left",
        )
        self._you_label.pack(fill=tk.X, pady=(0, 6))

        # "PAI:" label
        self._pai_label = tk.Label(
            frame,
            text="PAI: ...",
            font=("Segoe UI", 10),
            fg="#a6e3a1",
            bg="#1e1e2e",
            anchor="w",
            wraplength=390,
            justify="left",
        )
        self._pai_label.pack(fill=tk.X)

        # Start polling the queue
        self._poll_queue()

        # Run mainloop (blocks until window is destroyed)
        root.mainloop()

    def _poll_queue(self) -> None:
        """Drain pending updates and reschedule the next poll.

        Runs on the tkinter thread. Applies every queued conversation pair to
        the labels; on receiving the ``None`` sentinel it destroys the window
        and stops rescheduling. Otherwise it re-arms itself via
        ``root.after(100, ...)``.

        Side Effects:
            Mutates label text and schedules the next 100ms poll, or destroys
            the root window on shutdown.
        """
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg is None:
                    # Shutdown sentinel
                    self._root.destroy()
                    return
                user_text, pai_text = msg
                self._you_label.config(text=f"You: {user_text}")
                self._pai_label.config(text=f"PAI: {pai_text}")
        except queue.Empty:
            pass

        # Schedule next poll
        if self._root:
            self._root.after(100, self._poll_queue)
