# Command Lines for Linux:
ls : print all files and folders in the directory
mkdir <name_directory> : creates a new folder
touch <file name.extension> : creates a new file
cd <directory name> : changes directory (folder) <space must be "\ ">
cd .. : goes back one folder
cd ../.. : goes back two folders

# Command Lines for GitHub:
git clone <url link> : links VS Code to Github
git status : identifies if there anything from hard drive file to go to GitHub ("working tree clean")
git add "name of file" : adds file to queue for GitHub queue process
git add . : updates all files from hard drive for GitHub queue process
git commit -m "description" : commits new files in the GitHub queue
git push : updates local instance to global instance (repository in GitHub)
git pull : updates global instance to local instance 
git fetch : Updates local snapshot of GitHub
git diff origin/main : compares difference of local instance to the global instance, origin/main is the local snapshot of cloud
git commit -a -m "Message to describe changes" : combines both lines of git add and git commit
git log : provides list of commits from latest to oldest
q : quits the VIM code editor in the terminal
git log -p -1 : provides list of commits with their differences (-# = number of commits)