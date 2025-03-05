#!/usr/bin/env python3
"""
Script to continuously monitor and display the state of GPU servers, including user information.

Server entries in the server file should be provided as follows:
    10.150.52.155 -p80
    10.150.21.151 -p80
    10.150.20.213 -p22222

This script is particularly useful when used with an SSH key, so that a password is not required for each SSH connection.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import curses
from collections import defaultdict

# Default timeout (in seconds) after which SSH stops trying to connect
DEFAULT_SSH_TIMEOUT = 5

# Default timeout (in seconds) after which remote commands are interrupted
DEFAULT_CMD_TIMEOUT = 10

# Default server file
DEFAULT_SERVER_FILE = "serversV2.txt"
SERVER_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(sys.argv[0])), DEFAULT_SERVER_FILE
)

parser = argparse.ArgumentParser(description="Continuously check state of GPU servers")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
parser.add_argument(
    "-s", "--ssh-user", default=None, help="Username to use for SSH connections"
)
parser.add_argument(
    "--ssh-timeout",
    default=DEFAULT_SSH_TIMEOUT,
    help="Timeout (in seconds) after which SSH stops trying to connect",
)
parser.add_argument(
    "--cmd-timeout",
    default=DEFAULT_CMD_TIMEOUT,
    help="Timeout (in seconds) after which nvidia-smi is interrupted",
)
parser.add_argument(
    "--server-file", default=SERVER_FILE_PATH, help="File containing addresses of GPU servers"
)
parser.add_argument(
    "--refresh-interval", type=int, default=1, help="Refresh interval in seconds"
)
args = parser.parse_args()


def parse_server(server_str):
    """
    Parse a server string that includes an IP and an optional port in the format: "IP -pPORT".
    Returns a tuple (host, port) where port is None if not provided.
    """
    parts = server_str.split()
    host = parts[0]
    port = None
    for part in parts[1:]:
        if part.startswith("-p"):
            port = part[2:]
    return host, port


def build_ssh_command(server, remote_cmd):
    """
    Build an SSH command as a list by incorporating the host (and port if provided).
    If args.ssh_user is set, it prefixes the host.
    """
    host, port = parse_server(server)
    ssh_cmd = ["ssh"]
    if port:
        ssh_cmd.extend(["-p", port])
    if args.ssh_user:
        host = f"{args.ssh_user}@{host}"
    ssh_cmd.append(host)
    ssh_cmd.append(remote_cmd)
    return ssh_cmd


def run_nvidiasmi_local():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-q", "-x"], timeout=args.cmd_timeout
        )
    except subprocess.TimeoutExpired:
        logging.error("nvidia-smi command timed out")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"nvidia-smi command failed: {e}")
        return None


def run_nvidiasmi_remote(server, ssh_timeout, cmd_timeout):
    remote_cmd = f"timeout {cmd_timeout} nvidia-smi -q -x"
    ssh_cmd = build_ssh_command(server, remote_cmd)
    try:
        return subprocess.check_output(ssh_cmd, timeout=ssh_timeout)
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout while connecting to {server}")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running nvidia-smi on {server}: {e}")
        return None


def get_gpu_infos(nvidiasmi_output):
    if nvidiasmi_output is None:
        return []
    gpu_infos = []
    root = ET.fromstring(nvidiasmi_output)
    for gpu in root.findall("gpu"):
        gpu_info = {
            "index": gpu.find("minor_number").text,
            "name": gpu.find("product_name").text,
            "memory_total": int(gpu.find("fb_memory_usage/total").text.split()[0]),
            "memory_used": int(gpu.find("fb_memory_usage/used").text.split()[0]),
            "memory_free": int(gpu.find("fb_memory_usage/free").text.split()[0]),
            "processes": [],
        }
        for process in gpu.findall("processes/process_info"):
            pid = process.find("pid").text
            used_memory = int(process.find("used_memory").text.split()[0])
            gpu_info["processes"].append({"pid": pid, "used_memory": used_memory})
        gpu_infos.append(gpu_info)
    return gpu_infos


def get_user_info(server, pids):
    if not pids:
        return {}
    remote_cmd = f"ps -o user= -p {','.join(pids)}"
    # For local host, run the command directly
    if server.split()[0] in [".", "localhost", "127.0.0.1"]:
        try:
            output = subprocess.check_output(remote_cmd, shell=True, timeout=args.cmd_timeout)
            output = output.decode().strip()
            users = output.split("\n")
            return dict(zip(pids, users))
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out on {server}")
            return {}
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting user info on {server}: {e}")
            return {}
    else:
        ssh_cmd = build_ssh_command(server, remote_cmd)
        try:
            output = subprocess.check_output(ssh_cmd, timeout=args.cmd_timeout)
            output = output.decode().strip()
            users = output.split("\n")
            return dict(zip(pids, users))
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out on {server}")
            return {}
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting user info on {server}: {e}")
            return {}


def display_gpu_infos(stdscr, server, gpu_infos, col, row_offset):
    try:
        stdscr.addstr(row_offset, col, f"Server: {server}")
    except curses.error:
        return row_offset
    row = row_offset + 1
    for gpu_info in gpu_infos:
        if row >= curses.LINES - 1:
            break
        try:
            stdscr.addstr(row, col, f"GPU {gpu_info['index']} - {gpu_info['name']}")
            row += 1
            stdscr.addstr(row, col, f"  Memory Total: {gpu_info['memory_total']} MiB")
            row += 1
            stdscr.addstr(row, col, f"  Memory Used: ", curses.color_pair(1))
            stdscr.addstr(f"{gpu_info['memory_used']} MiB", curses.color_pair(2))
            row += 1
            stdscr.addstr(row, col, f"  Memory Free: ", curses.color_pair(1))
            stdscr.addstr(f"{gpu_info['memory_free']} MiB", curses.color_pair(3))
            row += 1

            # Display user information with yellow highlight
            pids = [process["pid"] for process in gpu_info["processes"]]
            user_info = get_user_info(server, pids)
            user_memory = defaultdict(int)
            for process in gpu_info["processes"]:
                user = user_info.get(process["pid"], "Unknown")
                user_memory[user] += process["used_memory"]

            for user, memory in user_memory.items():
                stdscr.addstr(row, col, "  User: ", curses.color_pair(1))
                stdscr.addstr(f"{user}", curses.color_pair(4))
                stdscr.addstr(", Memory Used: ", curses.color_pair(1))
                stdscr.addstr(f"{memory} MiB", curses.color_pair(4))
                row += 1

            row += 1
        except curses.error:
            break
    if row < curses.LINES - 1:
        try:
            stdscr.addstr(row, col, "-" * (curses.COLS // 2 - 1))
        except curses.error:
            pass
    row += 2
    return row


def main(stdscr, args):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Yellow text

    if os.path.exists(args.server_file):
        with open(args.server_file) as f:
            servers = [line.strip() for line in f if line.strip()]
    else:
        logging.error(f"Server file {args.server_file} does not exist.")
        return

    while True:
        stdscr.clear()
        max_row, max_col = stdscr.getmaxyx()
        col_width = max_col // 2
        left_col_offset = 0
        right_col_offset = 0

        for i, server in enumerate(servers):
            if i % 2 == 0:
                col = 0
                row_offset = left_col_offset
                nvidiasmi_output = (
                    run_nvidiasmi_local()
                    if server.split()[0] in [".", "localhost", "127.0.0.1"]
                    else run_nvidiasmi_remote(server, args.ssh_timeout, args.cmd_timeout)
                )
                left_col_offset = display_gpu_infos(
                    stdscr, server, get_gpu_infos(nvidiasmi_output), col, row_offset
                )
            else:
                col = col_width + 1  # Adjusted for vertical separator
                row_offset = right_col_offset
                nvidiasmi_output = (
                    run_nvidiasmi_local()
                    if server.split()[0] in [".", "localhost", "127.0.0.1"]
                    else run_nvidiasmi_remote(server, args.ssh_timeout, args.cmd_timeout)
                )
                right_col_offset = display_gpu_infos(
                    stdscr, server, get_gpu_infos(nvidiasmi_output), col, row_offset
                )

        # Draw vertical separator
        for row in range(max_row):
            try:
                stdscr.addch(row, col_width, "|")
            except curses.error:
                pass

        stdscr.refresh()
        time.sleep(args.refresh_interval)


if __name__ == "__main__":
    curses.wrapper(main, args)
