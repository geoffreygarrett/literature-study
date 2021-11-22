\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{tudat}[19/11/2021 Geoffrey H. Garrett, version 1.0]

\RequirePackage{xcolor}     % Using colours in documents
%\RequirePackage{color}     % Using colours in documents
\RequirePackage{tikz}       % Create graphic elements
\usetikzlibrary{positioning,arrows.meta,calc,decorations.pathreplacing}

%\RequirePackage{amsmath}    % Mathematics
%\RequirePackage{amssymb}    % Symbols
%\RequirePackage{siunitx}    % Various functions, e.g. \num{}
%
%\RequirePackage{tabularx}   % Additional functions to tables
%\RequirePackage{booktabs}   % Adds more line functionality to tables
%\RequirePackage{longtable}  % Lets tables break over page
%\RequirePackage{multirow}   % Counterpart of multi columns
%\RequirePackage{enumitem}   % Customise the list spacing
%
%\RequirePackage{geometry}   % Document geometry
%\RequirePackage{titlesec}   % Custom titles  % NO NO NO
%\RequirePackage{titletoc}   % Custom table of contents
%\RequirePackage{fancyhdr}   % Custom header/footer
\RequirePackage[hidelinks]{hyperref} % Improved referencing/links
%
\RequirePackage{graphicx}   % Adding images
%\RequirePackage{float}      % Additional float parameters
%\RequirePackage[labelfont=bf,justification=centering,footnotesize]{caption} % Captions
%\RequirePackage{subcaption} % Subfigures and subcaptions
%\RequirePackage{xcolor}     % Using colours in documents
%\RequirePackage{tikz}       % Create graphic elements
%
%% -> BibLaTeX moved to report.tex ->
%\RequirePackage{datetime}   % Used in preface for monthname
%\RequirePackage{microtype}  % Refinements towards typographical perfection
%\RequirePackage[nottoc]{tocbibind} % Add the lists to the table of contents
%\RequirePackage{xspace}     % Ensures correct spacing after macros like \deg
%\RequirePackage{etoolbox}   % General toolbox (e.g. \ifdefvoid)
\RequirePackage[T1]{fontenc}  % AIAA font

% makecover
\RequirePackage{graphicx}
\RequirePackage{tikz}
%\RequirePackage{xspace}

% double spacing
\newif\if@doublespace
\DeclareOption{doublespace}{\@doublespacetrue}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new-aiaa.cls, v1.2, 2018/01/10
% Developed by Overleaf
%
% This class file enables authors to prepare papers
% for submission to AIAA Technical Journals and
% Conferences.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ExecuteOptions{}
%\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[10pt,oneside]{book}


% Optional double spacing for draft/review.
\RequirePackage{setspace}
\if@doublespace
\doublespacing
\fi

% title sec
\RequirePackage[explicit]{titlesec}
\renewcommand{\thechapter}{\arabic{chapter}}
\renewcommand{\thesection}{\thechapter.\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Fancy chapters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
\makeatletter
    % see https://www.overleaf.com/learn/latex/sections_and_chapters
    \titlespacing
        {\chapter}                        % command
        {0pt}                             % left
        {-4ex}                            % before-sep
        {6ex}                             % after-sep
    \titleformat
        {\chapter}                        % command - the sectioning command to be redefined
        [frame]                           % shape - is sectioning paragraph shape; possible values are: hang, block, display, runin, leftmargin, rightmargin, drop, wrap, frame
        {\bfseries\large\rm}              % format -  is the format to be applied to the title, label, and text
        {\chaptername \ \thechapter}      % label - specify sectioning label
        {2.5ex}                           % sep - horizontal separation between label and title body
        {\huge\bfseries\centering{#1}}    % before-code - code preceding the title body
        []                                % after-code - code following the title body
\makeatother

% section
\titleformat{\section}
    {\large\bfseries\singlespacing\centering}
    {\thesection.\space}{0pt}{#1}[]
    \titlespacing{\section}{0pt}{0.5\baselineskip}{0pt}

% subsections
\titleformat{\subsection}
    {\normalsize\bfseries\singlespacing}
    {\Alph{subsection}.\space}{0pt}{#1}[]
    \titlespacing{\subsection}{0pt}{0.5\baselineskip}{0pt}

% subsubsections
\titleformat{\subsubsection}
    {\normalsize\itshape\singlespacing}
    {\arabic{subsubsection}.\space}{0pt}{#1}[]
    \titlespacing{\subsubsection}{0pt}{0.5\baselineskip}{0pt}

\setlength{\abovedisplayskip}{\baselineskip}
\setlength{\belowdisplayskip}{\baselineskip}
\setlength{\abovedisplayshortskip}{\baselineskip}
\setlength{\belowdisplayshortskip}{\baselineskip}


%% Some of the TU Delft house style colours
\definecolor{tudelft-cyan}{cmyk}{1,0,0,0}
\definecolor{tudelft-black}{cmyk}{0,0,0,1}
\definecolor{tudelft-white}{cmyk}{0,0,0,0}


%% Commands to define the title, author, etc
\renewcommand{\title}[1]{%
    \def\atitle{#1}%
    \hypersetup{pdftitle=#1}} % Adding to metadata

\renewcommand*{\author}[2][]{%
    \def\aauthor{#2}%
    \def\aauthor@short{#1}%
    \hypersetup{pdfauthor=\ifdefvoid{\aauthor@short}{#2}{#1}}} % Adding to metadata

%% Arguments to define
\newcommand*\subtitle[1]{\def\asubtitle{#1}}
\newcommand*\coverimage[1]{\def\@cover@image{#1}}
\newcommand\subject[1]{\def\asubject{#1}}

\definecolor{title}{HTML}{4884d6} % Color for the title
\usepackage{etoolbox}   % General toolbox (e.g. \ifdefvoid)
\usepackage{fontspec}

\newcommand*\makecover{
    \clearpage
    \thispagestyle{empty}
    \usetikzlibrary{positioning}

% cover image.
    \begin{tikzpicture}[remember picture, overlay]

        % cover image.
        \node[inner sep=0pt] at (current page.center) {%
            \includegraphics[height=\paperheight]{cover.png}%
        };

        \node[rotate=90,below right=4cm and .3cm] at (current page.west) {%
            \titlestyle\color{tudelft-white} Delft University of Technology};

        % white TU Delft logo.
        \node[above right=0.2cm and 0.6cm] at (current page.south west) {%
            \includegraphics[width=0.35\linewidth]{tudelft/logo-white.pdf}};


        \node[below=1cm,minimum width={\paperwidth},inner ysep=25pt,opacity=0.7,text opacity=1] at (current page.north) {%
            \begin{minipage}{0.9\paperwidth}
                \fontsize{35}{35}\selectfont\color{title}\centering\atitle \\[0.5ex]
                \ifdefvoid{\asubtitle}{}{\fontsize{22}{22}\selectfont\color{tudelft-white} The balance of risk and reward  \\[2.5ex]}
                \ifdefvoid{\asubject}{}{\fontsize{22}{22}\selectfont\color{tudelft-white} \\[0.5ex]}
                \fontsize{24}{24}\selectfont\color{tudelft-white}\centering Geoffrey H. Garrett
            \end{minipage}};

    \end{tikzpicture}
    \cleardoublepage{}
}