# Pool querying script

## Install

- Login to the tfpool login server
- Setup SSH keys such that you can login between pools without password authentication (see FAQ.md or below)
- Download the scripts and put them into some folder e.g. `~/poolscripts`
- Make them executable: `chmod u+x ~/poolscripts/*.sh`

## Usage

```bash
cd ~/poolscripts

# load / read infos. wait several minutes to complete! 
./pload.sh -v

# to force update if it says "not updating"
./pload.sh -v -f

# if you are impatient check output of file ./_logs/poolout.log
```

## Troubleshooting

### SSH does not trust the pools

If you get this error:

```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

You can fix the error below by forcibly adding all tfpools to known hosts with:

```bash
for i in $(seq 1 70); do foo=$(printf "%02d" ${i}); ssh-keygen -f "/home/USERNAME/.ssh/known_hosts" -R tfpool${foo} ; done
```

### Some single pool still requires password, even though SSH keys have been set properly and all other pools don't require password

You can exclude this pool by editing `constants.sh` variable `VALID_POOLS`

### How to setup SSH access from pool PCs to other pool PCs

- Login to the tfpool login server
- Create ssh key with command `ssh-keygen`, no passphrase. By default this will create file `~/.ssh/id_rsa`
- Copy the key on to the pool server (confirm with password): `ssh-copy-id -i ~/.ssh/id_rsa -o PubkeyAuthentication=no username@login.informatik.uni-freiburg.de`
- Edit/create file `.ssh/config` and add the lines:

~~~bash
Host tfpool*
    Port 22
    User username
    IdentityFile ~/.ssh/id_rsa
~~~
