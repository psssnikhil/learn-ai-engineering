/* MathJax configuration for MkDocs Material + pymdownx.arithmatex */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["ams", "boldsymbol"] },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

/* Re-typeset math after instant navigation (same pattern as mermaid-init.js). */
document$.subscribe(function () {
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    MathJax.typesetClear();
    MathJax.typesetPromise();
  }
});
