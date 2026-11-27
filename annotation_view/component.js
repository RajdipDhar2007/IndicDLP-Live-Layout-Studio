(function () {
  const root = document.getElementById("root");
  const send = (type, fields) => window.parent.postMessage({
    isStreamlitMessage: true,
    type,
    ...(fields || {})
  }, "*");

  let shown = new Set();  // ticks already on screen: only a new one plays the pop animation

  function render(event) {
    const message = event.data;
    if (!message || message.type !== "streamlit:render") return;

    const args = message.args || {};
    const width = Number(args.image_width);
    const height = Number(args.image_height);
    const detections = Array.isArray(args.detections) ? args.detections : [];
    // The page image is kept between renders (the OCR progress re-renders this view every second); only the boxes are rebuilt.
    let page = root.querySelector(".page");
    let image = page && page.querySelector(".page-image");
    const src = `data:image/jpeg;base64,${args.image_b64 || ""}`;
    if (!page) {
      page = document.createElement("div");
      page.className = "page";
      image = document.createElement("img");
      image.className = "page-image";
      image.alt = "Document page with clickable detected regions";
      page.appendChild(image);
      root.replaceChildren(page);
    }
    page.querySelectorAll(".region").forEach((el) => el.remove());
    page.classList.toggle("dense", detections.length > 30);
    if (image.dataset.src !== src) {
      image.dataset.src = src;
      image.src = src;
    }

    const shownNow = new Set();
    detections.forEach((detection, index) => {
      const [x1, y1, x2, y2] = detection.box.map(Number);
      const region = document.createElement("button");
      region.type = "button";
      region.className = "region";
      if (detection.label === "paragraph") region.classList.add("paragraph");
      if (y1 / height < 0.025) region.classList.add("top-edge");
      region.style.left = `${x1 / width * 100}%`;
      region.style.top = `${y1 / height * 100}%`;
      region.style.width = `${(x2 - x1) / width * 100}%`;
      region.style.height = `${(y2 - y1) / height * 100}%`;
      region.style.setProperty("--region-color", detection.color);
      region.title = `#${index + 1} ${detection.name} · ${(detection.conf * 100).toFixed(0)}%`;

      if (detection.transcribed || detection.ocr_failed) {
        const tick = document.createElement("span");
        tick.className = "region-tick" + (shown.has(`${index}:${detection.transcribed}`) ? "" : " new");
        shownNow.add(`${index}:${detection.transcribed}`);
        tick.textContent = detection.transcribed ? "✅" : "❌";
        tick.setAttribute("aria-label", detection.transcribed ? "OCR complete" : "OCR failed");
        region.appendChild(tick);
      }

      const label = document.createElement("span");
      label.className = "region-label";
      label.textContent = detection.name;
      region.appendChild(label);
      region.addEventListener("click", () => send("streamlit:setComponentValue", {
        dataType: "json",
        value: { index, nonce: `${Date.now()}-${Math.random()}` }
      }));
      page.appendChild(region);
    });

    shown = shownNow;
    const setHeight = () => send("streamlit:setFrameHeight", { height: Math.ceil(page.getBoundingClientRect().height + 4) });
    if (image.complete && image.naturalWidth) setHeight();
    else image.addEventListener("load", setHeight, { once: true });
  }

  window.addEventListener("message", render);
  send("streamlit:componentReady", { apiVersion: 1 });
})();
