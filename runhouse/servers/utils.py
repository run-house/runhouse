import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


LOCALHOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 32300
DEFAULT_HTTP_PORT = 80
DEFAULT_HTTPS_PORT = 443
LOCAL_HOSTS = ["localhost", LOCALHOST]
SERVER_LOGFILE = os.path.expanduser("~/.rh/server.log")
SERVER_START_CMD = f"{sys.executable} -m runhouse.servers.http.http_server"
SERVER_STOP_CMD = f'pkill -f "{SERVER_START_CMD}"'
# 2>&1 redirects stderr to stdout
START_SCREEN_CMD = f"screen -dm bash -c \"{SERVER_START_CMD} 2>&1 | tee -a '{SERVER_LOGFILE}' 2>&1\""
RH_RAY_PORT = 6379
# RAY_BOOTSTRAP_FILE = "~/ray_bootstrap_config.yaml"
# --autoscaling-config=~/ray_bootstrap_config.yaml
# We need to use this instead of ray stop to make sure we don't stop the SkyPilot ray server,
# which runs on other ports but is required to preserve autostop and correct cluster status.
RAY_START_CMD = f"ray start --head --port {RH_RAY_PORT}"
RAY_KILL_CMD = f'pkill -f ".*ray.*{RH_RAY_PORT}.*"'


def _add_flags_to_commands(flags, start_screen_cmd, server_start_cmd):
    flags_str = "".join(flags)

    start_screen_cmd = start_screen_cmd.replace(
        server_start_cmd, server_start_cmd + flags_str
    )
    server_start_cmd += flags_str

    return start_screen_cmd, server_start_cmd


def _start_server_cmds(
    restart,
    restart_ray,
    screen,
    create_logfile,
    host,
    port,
    use_https,
    den_auth,
    ssl_keyfile,
    ssl_certfile,
    restart_proxy,
    use_nginx,
    certs_address,
    use_local_telemetry,
):
    cmds = []
    if restart:
        cmds.append(SERVER_STOP_CMD)
    if restart_ray:
        cmds.append(RAY_KILL_CMD)
        # TODO Add in BOOTSTRAP file if it exists?
        cmds.append(RAY_START_CMD)

    server_start_cmd = SERVER_START_CMD
    start_screen_cmd = START_SCREEN_CMD

    flags = []

    den_auth_flag = " --use-den-auth" if den_auth else ""
    if den_auth_flag:
        logger.info("Starting server with Den auth.")
        flags.append(den_auth_flag)

    restart_proxy_flag = " --restart-proxy" if restart_proxy else ""
    if restart_proxy_flag:
        logger.info("Reinstalling Nginx and server configs.")
        flags.append(restart_proxy_flag)

    use_nginx_flag = " --use-nginx" if use_nginx else ""
    if use_nginx_flag:
        logger.info("Configuring Nginx on the cluster.")
        flags.append(use_nginx_flag)

    ssl_keyfile_flag = f" --ssl-keyfile {ssl_keyfile}" if ssl_keyfile else ""
    if ssl_keyfile_flag:
        logger.info(f"Using SSL keyfile in path: {ssl_keyfile}")
        flags.append(ssl_keyfile_flag)

    ssl_certfile_flag = f" --ssl-certfile {ssl_certfile}" if ssl_certfile else ""
    if ssl_certfile_flag:
        logger.info(f"Using SSL certfile in path: {ssl_certfile}")
        flags.append(ssl_certfile_flag)

    # Use HTTPS if explicitly specified or if SSL cert or keyfile path are provided
    https_flag = (
        " --use-https" if use_https or (ssl_keyfile or ssl_certfile) else ""
    )
    if https_flag:
        logger.info("Starting server with HTTPS.")
        flags.append(https_flag)

    host_flag = f" --host {host}" if host else ""
    if host_flag:
        logger.info(f"Using host: {host}.")
        flags.append(host_flag)

    port_flag = f" --port {port}" if port else ""
    if port_flag:
        logger.info(f"Using port: {port}.")
        flags.append(port_flag)

    address_flag = f" --certs-address {certs_address}" if certs_address else ""
    if address_flag:
        logger.info(f"Server public IP address: {certs_address}.")
        flags.append(address_flag)

    use_local_telemetry_flag = (
        " --use-local-telemetry" if use_local_telemetry else ""
    )
    if use_local_telemetry_flag:
        logger.info("Configuring local telemetry on the cluster.")
        flags.append(use_local_telemetry_flag)

    logger.info(
        f"Starting API server using the following command: {server_start_cmd}."
    )

    if flags:
        start_screen_cmd, server_start_cmd = _add_flags_to_commands(
            flags, start_screen_cmd, server_start_cmd
        )

    if screen:
        if create_logfile and not Path(SERVER_LOGFILE).exists():
            Path(SERVER_LOGFILE).parent.mkdir(parents=True, exist_ok=True)
            Path(SERVER_LOGFILE).touch()
        cmds.append(start_screen_cmd)
    else:
        cmds.append(server_start_cmd)

    return cmds
