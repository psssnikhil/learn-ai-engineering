/* Mermaid diagram initialization for MkDocs Material */
document$.subscribe(function () {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: false,
      theme:
        document.body.getAttribute("data-md-color-scheme") === "slate"
          ? "dark"
          : "default",
      securityLevel: "loose",
      flowchart: { useMaxWidth: true, htmlLabels: true },
      themeVariables: {
        primaryColor: "#3949ab",
        primaryTextColor: "#ffffff",
        primaryBorderColor: "#283593",
        lineColor: "#546e7a",
        secondaryColor: "#e8eaf6",
        tertiaryColor: "#f5f5f5",
      },
    });

    /* Re-run mermaid on every navigation (instant loading). */
    document.querySelectorAll(".mermaid").forEach(function (el) {
      const src = el.textContent || el.innerText;
      el.removeAttribute("data-processed");
      el.textContent = src;
    });

    mermaid.run({ querySelector: ".mermaid" });
  }
});
