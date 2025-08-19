// static/js/main.js
(function () {
  const btn = document.getElementById("downloadCsvBtn");
  const toast = document.getElementById("toast");
  const toggleTableBtn = document.getElementById("toggleTable");
  const tableDiv = document.getElementById("forecastTable");

  if (btn) {
    btn.addEventListener("click", function () {
      const b64 = this.getAttribute("data-csv-b64");
      const filename = this.getAttribute("data-filename") || "forecast.csv";
      if (!b64) return;

      const csvBytes = atob(b64);
      const u8 = new Uint8Array(csvBytes.length);
      for (let i = 0; i < csvBytes.length; i++) u8[i] = csvBytes.charCodeAt(i);

      const blob = new Blob([u8], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      if (toast) {
        toast.classList.add("toast-show");
        setTimeout(() => toast.classList.remove("toast-show"), 2500);
      }
    });
  }

  if (toggleTableBtn && tableDiv) {
    toggleTableBtn.addEventListener("click", () => {
      tableDiv.classList.toggle("d-none");
    });
  }
})();
