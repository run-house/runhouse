import threading
import queue

from runhouse.shell_handler import ShellHandler


def console(q, lock):
    while 1:
        with lock:
            cmd = input('>> ')

        q.put(cmd)
        if exit_cmd(cmd):
            break


def exit_cmd(cmd) -> bool:
    if cmd.lower() in ['quit', 'exit', 'quit()', 'exit()']:
        return True
    return False


def process_cmd_commands(sh: ShellHandler):
    cmd_queue = queue.Queue()
    stdout_lock = threading.Lock()

    dj = threading.Thread(target=console, args=(cmd_queue, stdout_lock))
    dj.start()

    while 1:
        cmd = cmd_queue.get()
        if exit_cmd(cmd):
            break
        # TODO maybe validate the user input? (check syntax errors, etc.)
        sh.execute(cmd)
