import threading
import queue
import typer

from runhouse.shell_handler import ShellHandler


def console(q, lock):
    typer.echo("****To exit at anytime type 'exit' or 'quit'****")
    while 1:
        typer.echo("Press Enter to begin writing commands")
        input()  # After pressing Enter you'll be in "input mode"
        with lock:
            cmd = input('>> ')

        q.put(cmd)
        if exit_cmd(cmd):
            break


def invalid_input(lock):
    with lock:
        print('--> Unknown command')


def exit_cmd(cmd) -> bool:
    if cmd.lower() == 'quit' or cmd.lower() == 'exit':
        return True
    return False


def cmd_commands(sh: ShellHandler):
    cmd_queue = queue.Queue()
    stdout_lock = threading.Lock()

    dj = threading.Thread(target=console, args=(cmd_queue, stdout_lock))
    dj.start()

    while 1:
        cmd = cmd_queue.get()
        if exit_cmd(cmd):
            break
        print("cmd", cmd)
        sh.execute(cmd)
