# progress_utils.py (Py 3.8+ 호환)
import sys, time, threading

class Heartbeat:
    """
    콘솔 한 줄에 스피너를 0.3~0.5초 주기로 갱신해 '살아있음'을 보여줍니다.
    with Heartbeat("메시지") 블록을 쓰거나 수동으로 enter/exit를 호출하세요.
    """
    def __init__(self, label="working...", interval=0.5):
        self.label = label
        self.interval = max(0.1, float(interval))
        self._stop = False
        self._t = None
        self._frames = "|/-\\"

    def _run(self):
        i = 0
        while not self._stop:
            sys.stdout.write("\r{0} {1}".format(self.label, self._frames[i % len(self._frames)]))
            sys.stdout.flush()
            i += 1
            time.sleep(self.interval)
        # 라인 지우고 개행
        sys.stdout.write("\r" + " " * (len(self.label) + 4) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        if self._t:
            self._t.join(timeout=0.2)

def print_step(msg):
    # 단계 시작/완료 표시용
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
