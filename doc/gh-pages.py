#!/usr/bin/env python
"""Script to commit the doc build outputs into the github-pages repo.

Use:

  gh-pages.py [tag]

If no tag is given, the current output of 'git describe' is used.  If given,
that is how the resulting directory will be named.

In practice, you should use either actual clean tags from a current build or
something like 'current' as a stable URL for the most current version of the """

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import os
import re
import shutil
import sys
from os import chdir as cd
from os.path import join as pjoin
from subprocess import PIPE, CalledProcessError, Popen, check_call

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

pages_dir = 'gh-pages'
html_dir = 'build/html'
pdf_dir = 'build/latex'
pdf_filename = 'scikit-rf.pdf'
copy_pdf = True
pages_repo = 'git@github.com:scikit-rf/doc.git'

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)


def sh2(cmd):
    """Execute command in a subshell, return stdout.

    Stderr is unbuffered from the subshell.x"""
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip()


def sh3(cmd):
    """Execute command in a subshell, return stdout, stderr

    If anything appears in stderr, print it out to sys.stderr"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()


def init_repo(path):
    """clone the gh-pages repo if we haven't already."""
    sh(f"git clone {pages_repo} {path}")
    here = os.getcwdu()
    cd(path)
    sh('git checkout gh-pages')
    cd(here)

#-----------------------------------------------------------------------------
# Script starts
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # The tag can be given as a positional argument
    try:
        tag = sys.argv[1]
    except IndexError:
        try:
            tag = sh2('git describe --exact-match')
        except CalledProcessError:
            tag = "dev"   # Fallback

    startdir = os.getcwdu()
    if not os.path.exists(pages_dir):
        # init the repo
        init_repo(pages_dir)
    else:
        # ensure up-to-date before operating
        cd(pages_dir)
        sh('git checkout gh-pages')
        sh('git pull')
        cd(startdir)

    dest = pjoin(pages_dir, tag)

    # don't `make html` here, because gh-pages already depends on html in Makefile
    # sh('make html')
    if tag != 'dev':
        # only build pdf for non-dev targets
        #sh2('make pdf')
        pass

    # This is pretty unforgiving: we unconditionally nuke the destination
    # directory, and then copy the html tree in there
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(html_dir, dest)
    if copy_pdf:
        shutil.copy(pjoin(pdf_dir, pdf_filename), pjoin(dest, pdf_filename))
        pass

    try:
        cd(pages_dir)
        status = sh2('git status | head -1')
        branch = re.match(r'\# On branch (.*)$', status).group(1)
        if branch != 'gh-pages':
            e = f'On {pages_dir}, git branch is {branch}, MUST be "gh-pages"'
            raise RuntimeError(e)

        sh(f'git add -A {tag}')
        sh(f'git commit -m"Updated doc release: {tag}"')
        print()
        print('Most recent 3 commits:')
        sys.stdout.flush()
        sh('git --no-pager log --oneline HEAD~3..')
    finally:
        cd(startdir)

    print()
    print(f'Now verify the build in: {dest!r}')
    print("If everything looks good, 'git push'")
