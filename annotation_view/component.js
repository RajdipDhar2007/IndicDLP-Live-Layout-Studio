(function () {
  const root = document.getElementById("root");
  const send = (type, fields) => window.parent.postMessage({
    isStreamlitMessage: true,
    type,
    ...(fields || {})
  }, "*");

  function render(event) {
    const message = event.data;
    if (!message || message.type !== "streamlit:render") return;

    const args = message.args || {};
    const width = Number(args.image_width);
    const height = Number(args.image_height);
    const detections = Array.isArray(args.detections) ? args.detections : [];
    root.replaceChildren();

    const page = document.createElement("div");
    page.className = "page";
    if (detections.length > 30) page.classList.add("dense");

    const image = document.createElement("img");
    image.className = "page-image";
    image.alt = "Document page with clickable detected regions";
    image.src = `data:image/jpeg;base64,${args.image_b64 || ""}`;
    page.appendChild(image);

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

      if (detection.transcribed) {
        const tick = document.createElement("span");
        tick.className = "region-tick";
        tick.textContent = "\u2713";
        tick.setAttribute("aria-label", "OCR complete");
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

    root.appendChild(page);
    const setHeight = () => send("streamlit:setFrameHeight", { height: Math.ceil(page.getBoundingClientRect().height + 4) });
    if (image.complete) setHeight();
    else image.addEventListener("load", setHeight, { once: true });
  }

  window.addEventListener("message", render);
  send("streamlit:componentReady", { apiVersion: 1 });
})();
