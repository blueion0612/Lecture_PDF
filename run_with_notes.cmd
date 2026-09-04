@echo off
rem Same as run.cmd, but keeps the lecturer's handwriting: one page per
rem write-and-wipe cycle, plus the clean slide underneath it.
call "%~dp0run.cmd" --ink-mode all
