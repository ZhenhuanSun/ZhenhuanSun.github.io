---
author: ["Zhenhuan Sun"]
title: 'Setting Up TeXstudio on macOS'
summary: "This guide walks through how to install and configure TeXstudio on macOS."
date: 2026-08-31
ShowToc: true
---

The slides I prepared for my Thesis Proposal Exam have grown to the point where Overleaf can no longer compile them within 
the time limit of the free plan. Instead of paying $12.43 a month to Overleaf just to extend compile time, I decided to 
set up a local LaTeX environment with MacTeX and TeXstudio for free instead. During the setup process, I ran into a few 
hiccups, so I figured I might as well documents them and the solutions I found for my future reference.

## Installation

MacTeX and TeXstudio play roles similar to a compiler and an IDE in programming. MacTeX provides the LaTeX compilers, such 
as `pdflatex`, to turn `.tex` source files into documents like PDFs, while TeXstudio provides a graphical editing environment 
for writing, compiling, and previewing those files. To get started, first download MacTex [here](https://www.tug.org/mactex/),
then download TexStudio [here](https://www.texstudio.org/).

## TexStudio `lastSession.txss2` Permission Error

At the time of writing this blog, I installed the latest version of TexStudio, version 4.9.7, on my M3 iMac. I noticed that
every time I closed TeXstudio, the following error message would appear

```text
Storing session information into /Users/zhenhuansun/.config/texstudio/lastSession.txss2 failed. File exists but is not writeable.
```

And when I `cd`'d into the `.config` directory in my home directory, I realized that there was no directory named `texstudio`.
At first, I tried to ignore the message because it did not seem to affect TeXstudio’s normal functionality. Changes to my 
files were saved properly, and I had no issues compiling my `.tex` files. However, I soon realized that this issue prevented 
TeXstudio from saving the customizations I made to its settings. For example, changes to the appearance and custom commands
would be lost after I closed and reopened the application. Fortunately, I was not the first person to encounter this issue, 
and a solution had already been posted [here](https://github.com/texstudio-org/texstudio/issues/3736). After reading the
solution and understanding what each command does, I came up with a slightly cleaner solution that proceeds as follows:

1.  In `~/.config` directory, create the `texstudio` directory by running

    ```bash
    sudo mkdir texstudio
    ```
    
    Because the directory is created with `sudo`, it will initially be owned by `root`. You can verify this by running
    `ls -ld texstudio`, which should return something like

    ```bash
    drwxr-xr-x  2 root  wheel  64 Aug 31 15:43 texstudio
    ```
    
    Here, `root` is the owner of the directory and `wheel` is its group. `drwxr-xr-x` can be broken down as `d | rwx | r-x | r-x`.
    From left to right, these parts indicate that `texstudio` is a directory, the owner has read, write and enter/traverse 
    permissions, the group has read and enter/traverse permissions, and everyone else also has read and enter/traverse permissions.
    Because only the owner has write permission, its ownership needs to be changed before TeXstudio can write to it.

2.  To change the ownership of the `texstudio` directory to your user account, run

    ```bash
    sudo chown username:staff texstudio
    ```
    
    Replace `username` with your username. Verify the change with `ls -ld texstudio`, you should see your username listed
    as the owner and `staff` as the group.

After these two steps, TeXstudio should be able to write into the `~/.config/texstudio` directory without triggering the 
permission error. As a result, closing TeXstudio should no longer produce the error message shown earlier, and any customizations 
you make should now be saved properly.

## Messy Build Files

After you finish editing your `.tex` files and click the `Build & View` button in TeXstudio, the compiler will generate 
a number of auxiliary build files, such as `.aux`, `.log`, `.toc`, `.out`, and `.synctex.gz` files, in the directory 
containing the `.tex` file being compiled, in addition to the final PDF. This can make the project directory very messy. 
To keep the project directory uncluttered, you can keep all generated files in a separate `build` directory. To do that,
first verify that your default compiler is PdfLaTeX by going to `Preferences -> Build` and checking `Default Compiler`,
then follow the following steps:

1.  Create a directory named `build` in your project directory.

2.  Navigate to `Preferences -> Commands`, add `-output-directory=build` to the PdfLaTeX command. The resulting PdfLaTeX 
    command should be something like

    ```bash
    pdflatex -synctex=1 -output-directory=build -interaction=nonstopmode %.tex
    ```
    
3.  Navigate to `Preferences -> Build` and check the `Show Advanced Options` box in the bottom-left corner. Then, under 
    `Additional Search Paths`, type `build` to both `Log File` and `PDF File` fields.

After these three steps, all files generated when building the `.tex` files will be stored in the `build` directory. However, 
this also introduced three new issues: 

1.  The `Files` panel on the left side of the interface kept pointing to my home directory instead of the current project 
    directory whenever I launched TeXstudio by opening a `.tex` file located in that project directory.
    
    -   This issue can be resolved by going to `Preferences -> General`, checking the `Show Advanced Options` box in the 
        bottom-left corner, and then unchecking the `Restore Previous Session at Startup` box. This forces TeXstudio to 
        start fresh, allowing the `Files` panel to point to the directory of the file you actually open. 

2.  The TeXstudio log showed the following error when I tried to build a `.tex` file, e.g., `main.tex`, that contained 
    references managed BibTeX.

    ```text
    Process started: bibtex "main".aux
    I couldn't open file name `main.aux'
    Process exited with error(s)
    Error: Command failed with error code 1
    ```
    
    As a result, all citations in the compiled PDF were displayed as `?`.

    -   This issue arises from BibTeX looking for `main.aux` in the project root directory instead of `build` directory.
        To resolve it, go to `Preferences -> Commands`, change `bibtex %.aux` in the BibTeX command to `bibtex build/%.aux`,
        so that BibTeX looks for the `.aux` file in the `build` directory.

3.  The TeXstudio log showed the following error when I tried to open the compiled PDF in an external window.
    
    ```text
    Process started: open "main".pdf
    The file /your_project_directory/main.pdf does not exist.
    Process exited with error(s)
    ```    

    -   This issue arises due to the same reason as the previous issue and shares a similar solution. Go to `Preferences -> Commands`,
        change `open %.pdf > /dev/null` in the External PDF Viewer command to `open build/%.pdf > /dev/null`, so that TeXstudio
        looks for the `.pdf` file in the `build` directory.

Also see [this documentation](https://texstudio-org.github.io/configuration.html) for more information on configuring TeXstudio.

## Useful Tricks

### `Command + left-click`

I have always found Overleaf's feature of jumping to the corresponding location in the `.tex` source by double-clicking
a word in the compiled PDF very useful. TeXstudio offers a similar, but more convenient, feature. To jump from a word in
the compiled PDF to its corresponding location in the `.tex` source, `Command + left-click` that word in the compiled PDF.
TeXstudio also supports the reverse. To jump from a piece of text in the `.tex` source to its corresponding location in 
the compiled PDF, `Command + left-click` that text in the `.tex` source.
