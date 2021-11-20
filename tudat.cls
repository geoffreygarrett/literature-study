\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{tudat}[19/11/2021 Geoffrey H. Garrett, version 1.0]

\RequirePackage{xcolor}     % Using colours in documents
\RequirePackage{color}     % Using colours in documents
\RequirePackage{tikz}       % Create graphic elements
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
\RequirePackage{titlesec}   % Custom titles
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
\RequirePackage{etoolbox}   % General toolbox (e.g. \ifdefvoid)
\RequirePackage[T1]{fontenc}

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


%%%%% 1. General Definitions for the Document and Bibliography %%%%%

%% Commands to define the title, author, etc
\renewcommand{\title}[1]{%
    \def\@title{#1}%
    \hypersetup{pdftitle=#1}} % Adding to metadata

\renewcommand*{\author}[2][]{%
    \def\@author{#2}%
    \def\@author@short{#1}%
    \hypersetup{pdfauthor=\ifdefvoid{\@author@short}{#2}{#1}}} % Adding to metadata

\newcommand*\subtitle[1]{\def\@subtitle{#1}}
\newcommand*\coverimage[1]{\def\@cover@image{#1}}
\newcommand\subject[1]{\def\@subject{#1}}

%% Some of the TU Delft house style colours
\definecolor{tudelft-cyan}{cmyk}{1,0,0,0}
\definecolor{tudelft-black}{cmyk}{0,0,0,1}
\definecolor{tudelft-white}{cmyk}{0,0,0,0}

\newcommand*\makecover{
    % Clearing the page and removing page number
    \clearpage
    \thispagestyle{empty}

    %% Use the Tikz library positioning
    \usetikzlibrary{positioning}

    % Defining where everything needs to go
% Defining where everything needs to go
    \begin{tikzpicture}[overlay,remember picture]

        \node[above=0,inner sep=0] at (current page.south) {\includegraphics[width=\paperwidth]{\@cover@image}};

        \node[rotate=90,below right=4cm and .3cm] at (current page.west) {%
            \titlestyle\color{white} Delft University of Technology};

        \node[above right=0.2cm and 0.6cm] at (current page.south west) {%
            \includegraphics[width=0.35\linewidth]{layout/tudelft/logo-white.pdf}};

        \node[below=2cm,fill=tudelft-black,minimum width={\paperwidth},inner ysep=25pt,opacity=0.7,text opacity=1] at (current page.north) {%
            \begin{minipage}{0.9\paperwidth}
                \largetitlestyle\fontsize{50}{50}\selectfont\color{title}\@title \\[0.5ex]
                \ifdefvoid{\@subtitle}{}{\titlestyle\fontsize{22}{22}\selectfont\color{white}\@subtitle \\[2.5ex]}
                \ifdefvoid{\@subject}{}{\titlestyle\fontsize{22}{22}\selectfont\color{white}\@subject \\[0.5ex]}
                \largetitlestyle\fontsize{24}{24}\selectfont\color{white}\@author
            \end{minipage}};

    \end{tikzpicture}
    \newpage
}

%\RequirePackage[indentfirst=false,font+=small,leftmargin=0.4in,rightmargin=0pt,vskip=0pt]
%{quoting}