(function () {
  const navToggle = document.querySelector("[data-nav-toggle]");
  const siteNav = document.querySelector("[data-site-nav]");

  if (navToggle && siteNav) {
    navToggle.addEventListener("click", () => {
      const expanded = navToggle.getAttribute("aria-expanded") === "true";
      navToggle.setAttribute("aria-expanded", String(!expanded));
      siteNav.classList.toggle("is-open", !expanded);
    });

    siteNav.addEventListener("click", (event) => {
      if (event.target instanceof HTMLAnchorElement) {
        const targetId = event.target.getAttribute("href");
        navToggle.setAttribute("aria-expanded", "false");
        siteNav.classList.remove("is-open");

        if (targetId && targetId.startsWith("#")) {
          const target = document.querySelector(targetId);
          if (target) {
            event.preventDefault();
            window.requestAnimationFrame(() => {
              target.scrollIntoView({ block: "start" });
              history.pushState(null, "", targetId);
            });
          }
        }
      }
    });
  }

  const header = document.querySelector("[data-site-header]");
  const updateHeader = () => {
    if (header) {
      header.classList.toggle("is-scrolled", window.scrollY > 12);
    }
  };
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.getAttribute("data-copy-target");
      const target = targetId ? document.getElementById(targetId) : null;
      if (!target) {
        return;
      }
      const original = button.textContent;
      try {
        await navigator.clipboard.writeText((target.textContent || "").trim());
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select";
      }
      window.setTimeout(() => {
        button.textContent = original;
      }, 1600);
    });
  });

  document.querySelectorAll("[data-tabs]").forEach((tabs) => {
    const tabButtons = Array.from(tabs.querySelectorAll("[data-tab]"));
    const panels = Array.from(tabs.querySelectorAll("[data-panel]"));
    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const name = button.dataset.tab;
        tabButtons.forEach((item) => {
          const active = item === button;
          item.classList.toggle("is-active", active);
          item.setAttribute("aria-selected", String(active));
        });
        panels.forEach((panel) => {
          const active = panel.dataset.panel === name;
          panel.classList.toggle("is-active", active);
          panel.hidden = !active;
        });
      });
    });
  });
})();
