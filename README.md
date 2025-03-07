# Remote GPU Monitor

This Python script allows to check for free Nvidia GPUs in remote servers.
Additional features include to list the type of GPUs and who's using them.
The idea is to speed up the work of finding a free GPU in institutions that share multiple GPU servers.

The script works by using your account to SSH into the servers and running `nvidia-smi`. 

## Requirements

- python3
- SSH access to some Linux servers with Nvidia GPUs
- If the server you connect to uses a different user name than your local name, you either have to specify your name on the servers using the `-s` option, or set up access as described in [setup for convenience](#setup-for-convenience).

## Usage

For checking for free GPUs on some server(s), simply add their address(es) after the script name.
You might need to enter your password. To avoid that, follow the steps in [setup for convenience](#setup-for-convenience).

```
> ./gpu_monitor.py
```

If you have some set of servers that you regularily check, specify them in the file `servers.txt`, one address per line.
For example,
```
10.100.100.000 -p222
```
Once you did that, running just `./gpu_monitor.py` checks all servers specified in this file by default.

## Setup for Convenience

### Setting up an SSH key
If you want to avoid having to enter your password all the time, you can setup an SSH key to login into your server.
If you did this already, you are fine.

1. Open a terminal and run `cd .ssh`
2. Run `ssh-keygen` and follow the instructions.
It might be a good idea to not use the default file but to specify a specific filename reflecting the servers you are connecting to.
3. Run `ssh-copy-id -p<portnumber> <user>@<server>`, where `<user>@<server>` is the server you want to connect. If you chose a different filename for your key, you need to pass the filename with the `-i` option.
4. Enter the password of the target server.
5. Repeat step 3,4 for every server you want to connect to (not necessary if you have a shared home directory on all the servers).
6. Try to connect to the server using `ssh <user>@<server>`.
The first time you connect, it should ask you for the password of the SSH key.
If you are asked for the password multiple times, you might need to manually activate your SSH key using `ssh-add <path_to_ssh_key>`.
If it still does not work, follow with the next steps.
(Refer https://blog.naver.com/hanajava/220970169755)
### If you have a different user name on your local machine

This will show you how to avoid having to give your user name if you use the script (and SSH).

1. Go to the folder `.ssh` in your home and open the file `config`.
If it is not there, create it.
2. Add something like this:
```
Host 10.100.000.000 -p<portnumber>
User myusername
```
If you are connecting to multiple servers under the same domain, you can also use `Host *.mydomain.com` to indicate that you are using the same user name for all of them.
3. If you have an SSH key with a different name, you also add the line `IdentityFile path_to_ssh_key` after the `User` line.

