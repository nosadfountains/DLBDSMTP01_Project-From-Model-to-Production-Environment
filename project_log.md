# Project Log

## 2026-10-29
- **Did:** Created the GitHub repository and cloned it in VS Code
- **Problem:** "Publish to GitHub" button was not shown in VS Code; the Cmd+Shift+P shortcut did not work
- **Solution:** Created the repository on github.com first, then cloned it via View → Command Palette → "Git: Clone"
- **Hours:**

## 2026-11-02
- **Did:** Set up virtual environment, base packages, requirements.txt, folder structure, .gitignore, README and project log
- **Problem:** `python: command not found`
- **Solution:** The environment was not active; activated it with `source .venv/bin/activate`, later set it as the VS Code interpreter so it activates automatically
- **Problem:** Accidentally ran commands in a different project folder
- **Solution:** Checked the location with `pwd` before running commands
- **Problem:** A failed `pip freeze > requirements.txt` left an empty file
- **Learned:** `>` empties the target file before the command runs
- **Problem:** Sync to GitHub stopped working; `git status` reported "not a git repository"
- **Cause:** "Git: Clone" had created the repository as a subfolder; all work was being saved in the parent folder, which is not connected to GitHub
- **Solution:** Moved the repository to `~/Projects` (a path without spaces), moved the work files into it, recreated the virtual environment there
- **Problem:** The repository had no .gitignore, so the virtual environment could have been uploaded
- **Solution:** Added a .gitignore for `.venv/`, `data/`, model files, cache files, secrets and `.DS_Store`
- **Problem:** Test file name contained `#` and spaces, which breaks Python imports and terminal commands
- **Solution:** Deleted the file
- **Hours:1