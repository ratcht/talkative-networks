// Arithmatex (generic mode) emits math as \(...\) and \[...\] inside
// <script type="math/tex"> placeholders; this tells MathJax 3 to pick those up
// and re-typeset after MkDocs' instant navigation swaps page content.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};
