document.addEventListener("DOMContentLoaded", function () {
    console.log("Ransomware Detection Dashboard Loaded");
});
// ===============================
// AI Ransomware Dashboard JS
// ===============================

document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (!form) return;

    form.addEventListener("submit", function () {

        // Disable button
        const btn = document.querySelector("button[type='submit']");
        btn.disabled = true;
        btn.innerHTML = "⏳ Detecting...";

        // Progress Container
        let progressContainer = document.createElement("div");
        progressContainer.id = "progress-container";

        let progressBar = document.createElement("div");
        progressBar.id = "progress-bar";

        progressContainer.appendChild(progressBar);

        form.appendChild(progressContainer);

        let width = 0;

        let timer = setInterval(function () {

            width += 2;

            progressBar.style.width = width + "%";
            progressBar.innerHTML = width + "%";

            if (width >= 100) {

                clearInterval(timer);

                btn.innerHTML = "✔ Processing...";

            }

        }, 40);

    });

});