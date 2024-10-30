# Frequently Asked Questions.

## Setup:

- Read the setup instructions linked on the website and follow the installation closely

## Problems with installing conda

- Problem: **conda: Befehl nicht gefunden / Command not found**
- Check if you installed conda correctly as described in the setup
- Check your bashrc file: `nano ~/.bashrc`
  - Somewhere near the end you should find a block of code that starts with `# >>> conda initialize >>> ... `
  - If not, try the installation again and make sure to say "yes" when being asked to initialize your bashrc.
- If you find the code block in your bashrc, the problem is: .bashrc is not sourced when logging in via ssh (see [bashrc-at-ssh-login](https://stackoverflow.com/questions/820517/bashrc-at-ssh-login))
- Solution: if `~/.bash_profile` does not exist, create it and paste the following

```
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
```

- Problem: Conda runs out of space because you installed it into the wrong folder (home, "~" folder)
- Solution: Delete the existing installation and install it into the project folder

## Torch does not find the GPU

- On the tfpool install python 3.8 pytorch 1.13 cuda 11.6 (newer versions of cuda may not work)
- On your own PC, update the drivers and keep installing older versions of pytorch and cuda
  until you find one that works.

## Data access options

- work on pool pcs
- copy to local via vscode (right click -> download)
- use scp to download data from the tf server to your local machine
- mount the shared-data directory on your local machine using sshfs

## Problems with TF pool

- Check [TF pool FAQ](http://poolmgr.informatik.uni-freiburg.de/?id=103)
- Project folder is empty: There are invisible folders, use `cd` and the full path where you want to go instead of using `ls` or completion with the tab key.

### Which GPUs are free?

- Try [this website](https://tfpool.retool.com/embedded/public/79b8f1d7-c1bd-473c-a27f-4b7c6ab4693f)
  or consult the readme in `poolscripts/` to get a list of free GPUs. 

### Using jupyter notebooks on the pool

The easy way to do this is to run the server and browser on the same machine (either directly sit in the pool
or setup everything on your home computer).

However, if your code is running on a *different computer as the browser* , i.e.
you are running the code on a pool machine
but running the browser on your home computer) you will need to:

1. start the respective server on the pool machine.
2. establish a SSH tunnel from your home computer to the tf login node.
3. establish a SSH tunnel from the tf login node to the pool machine where the server is running.

The tunneling command looks like this:

`ssh -v -N -L ${port}:localhost:${port} "${user}@${targethost}"`

You might need to change the port of your server application if the port is already in use by another person.
Note that everyone who can access the login node will be potentially able to see your server.
Jupyter notebook protects this with an access token by default, but tensorboard does not.


### I have to input my password all the time

Create an ssh key on your local PC to login to the login server and create another key
on the login server to login to the pool PCs. [Tutorial](https://www.ssh.com/academy/ssh/keygen)

- Use command `ssh-keygen` to create keys
- Use command `ssh-copy-id` to copy keys to the target machines
- Edit the file `~/ssh/config` to configure which server to use which key.

### Access denied / Password issues

- Problem: login at [NextCloud](https://nc.informatik.uni-freiburg.de/index.php/apps/rainloop/) works but not when using ssh
- Solution: do not use Umlauts (äöüß etc) in you password. Different encodings in the browser (setting the password) and terminal lead to different hashes

### Home folder exceeds 1GB

- can lead to weird behaviour
- you cannot create new files
- list folder and file sizes in home: `cd ~`, `du --max-depth=1 -B M`
- you should be able to delete .cache (`rm -r .cache`) without negative impact
- also check your trash and .local and .share and your mailfolder if you can reduce them but be careful!
- **vscode-server can be large**
  - here is a nice solution to move it to your project space
  - it involves creating a symlink in your home directory
  - [move-vscode-server](https://stackoverflow.com/questions/62613523/how-to-change-vscode-server-directory)
- **pytorch pretrained models**
  - the pretrained weights are stored under `~/.cache/torch...`
  - change this directory in all python scripts
  - [see stackoverflow](https://stackoverflow.com/questions/52628270/is-there-any-way-i-can-download-the-pre-trained-models-available-in-pytorch-to-a)
- make sure to install conda into project folder and not home folder.

### Where is the physical location of the pool machines? (2023)

 It's in building 076 (old "Uniradio" site). In the room on the ground floor there are
20 seats with a PC each. The room on the upper floor is now intended mostly for group study and
has only 6 or 7 pool PCs available.

## Recommended: vscode - IDE for working on pool machines

### VS Code server setup

steps to install vs-code (on linux without sudo)

- download https://code.visualstudio.com/docs/?dv=linux64
- you can probably skip the next steps if you installed via the package manager
- copy to <path_to_bin_and_install>/install
- extract `tar -xf code-stable-x64-1636111355.tar.gz`
- make a bin directory next to install i.e. <path_to_bin_and_install>/bin: `mkdir bin`
- cd bin
- create softlink: `ln -s ../install/VSCode-linux-x64/bin/code vscode`
- add to your bash to call vscode from everywhere: `export PATH="<path_to_bin_and_install>/bin:$PATH"`

### VSCode - remote

- follow the steps in https://code.visualstudio.com/docs/remote/ssh
- open it with vscode from any terminal
- hit crtl p and paste `ext install ms-vscode-remote.vscode-remote-extensionpack`
- click install
- press f1 and click on add host
- enter the ssh command that you used from your local machine before to connect to the pool
- e.g. replace <username> with your username: ssh <username>@login.informatik.uni-freiburg.de
- then enter password, press f1 again and connect to host, and voila, you're on the tf pool
- click on open folder and enter the path to your project
- `/project/cv-ws2324/<username>/cv-exercises/`
- (in order to have the cv-exercises folder you need to do a `git clone https://github.com/lmb-freiburg/cv-exercises.git`)

### VSCode Shortcuts:

	- https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf
	- https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf
	- https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf
	- open terminal: ctrl j
	- open settings: ctrl ,
	- open command palette: ctrl shift p
	- set python interpreter of your conda environment: 
		ctrl shift p 
		python select interpreter
		choose path to your conda environment
		complaints about not knowing numpy will be gone

### Remote SSH in vscode directly to a pool machine

Instead of connecting vscode to the login node via ssh and then connect in the terminal to one of
the pool machines, one can directly connect via ssh to a pool machine as follows:
- Crtl+shift+p: then Remote-SSH: Open SSH Configuration File and choose your loca config file.
- Usually you will see an entry for every server like:
```
Host login.informatik.uni-freiburg.de
  HostName login.informatik.uni-freiburg.de
  User username
```
Since the pool machines are only accessible after connecting to login, you can add one more entry for a pool machine as follows:
```
Host tfpool21
  HostName tfpool21
  ProxyJump login.informatik.uni-freiburg.de
  User username
```
Then, when you select tfpool21, it will first connect to login and then to tfpool21, hence will ask for the password twice.

This is required when running jupyter notebook that requires libraries that are not installed on the login node (exercise 07).

### Tensorboard visualization in vscode

To run tensorboard, you run the command from the terminal:

`tensorboard --logdir path`

This will show a link and probably a window will pop-up with 'open-in-local-browser'

if it does not pop up: hover over link in cmd and click on 'follow link using forwarded port'

## Additional Material

### Batch Norm

- nice explanation of intuition and parameters: [batchnorm-towardsdatascience](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)

## Exam

- Are the exercises relevant for the exam? Answer: The exam will be based on the lecture content. Nonetheless it is a lot easier to understand the concepts if you actually use them in practice.
- When is the exam? Answer: The exam will show up in HisInOne once it is planned.
- When is the exam review? Answer: The exam review date will be posted on the website. Typically it is at the very end of the semester break.

