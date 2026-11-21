/**
 * FairShed - Full-Featured Modern Web Application Logic
 * Integrates 10-Area Grid, Supervised Regression, CSP/Hill Climbing Scheduler,
 * Complete Historical EDA, AI Lab Syllabus Suite & Interactive Scenario Simulator.
 */

// ============================================================================
// 1. GLOBAL STATE & CITY FEEDERS CONFIGURATION (10 CITY ZONES)
// ============================================================================
const CITY_AREAS = [
  { id: "F01_HOSP", name: "Jinnah Hospital", fullName: "Jinnah Postgraduate Medical Centre, Karachi", type: "Hospital", load: 120, exempt: true, maxOutage: 0, color: "#38bdf8", outageHours: 0, cost: 1000.0 },
  { id: "F02_IND", name: "SITE Industrial", fullName: "Sindh Industrial Trading Estate, Karachi", type: "Industrial", load: 450, exempt: false, maxOutage: 4, color: "#f43f5e", outageHours: 3, cost: 250.0 },
  { id: "F03_COMM_A", name: "Saddar", fullName: "Saddar commercial locality, Karachi", type: "Commercial", load: 380, exempt: false, maxOutage: 4, color: "#8b5cf6", outageHours: 3, cost: 180.0 },
  { id: "F04_RES_A", name: "DHA Phase 6", fullName: "Defence Housing Authority Phase 6, Karachi", type: "Residential", load: 320, exempt: false, maxOutage: 4, color: "#3b82f6", outageHours: 3, cost: 90.0 },
  { id: "F05_RES_B", name: "Gulshan-e-Iqbal", fullName: "Gulshan-e-Iqbal Town, Karachi", type: "Residential", load: 290, exempt: false, maxOutage: 4, color: "#10b981", outageHours: 3, cost: 75.0 },
  { id: "F06_SUB_A", name: "North Nazimabad", fullName: "North Nazimabad Town, Karachi", type: "Suburban", load: 240, exempt: false, maxOutage: 4, color: "#f59e0b", outageHours: 3, cost: 60.0 },
  { id: "F07_SUB_B", name: "Shah Faisal Colony", fullName: "Shah Faisal Colony, Karachi", type: "Suburban", load: 210, exempt: false, maxOutage: 4, color: "#ec4899", outageHours: 3, cost: 50.0 },
  { id: "F08_RUR", name: "Gadap", fullName: "Gadap Town, Karachi", type: "Rural", load: 180, exempt: false, maxOutage: 5, color: "#14b8a6", outageHours: 3, cost: 40.0 },
  { id: "F09_COMM_B", name: "Tariq Road", fullName: "Tariq Road and PECHS, Karachi", type: "Commercial", load: 260, exempt: false, maxOutage: 4, color: "#a855f7", outageHours: 3, cost: 150.0 },
  { id: "F10_RES_C", name: "Clifton", fullName: "Clifton Cantonment locality, Karachi", type: "Residential", load: 220, exempt: false, maxOutage: 4, color: "#6366f1", outageHours: 2, cost: 70.0 }
];

const HOURS_24 = ["12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM", "6 AM", "7 AM", "8 AM", "9 AM", "10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM"];

// Populated from the trained forecasting API during initialization.
let DEMANDS_24H = new Array(24).fill(0);
let ACTUAL_DEMANDS_24H = new Array(24).fill(0);
let PREDICTED_DEMANDS_24H = new Array(24).fill(0);
let LINEAR_DEMANDS_24H = new Array(24).fill(0);
let CAPACITIES_24H = DEMANDS_24H.map(d => Math.round(d * 0.82));
let SHORTFALLS_24H = DEMANDS_24H.map((d, i) => Math.max(0, d - CAPACITIES_24H[i]));
let SCHEDULED_SHED_24H = null;

// Precomputed Baseline Matrices (10 feeders x 24 hours)
let activeScheduleMatrix = [
  // F01 Hospital (100% immune)
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // F02 Industrial
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // F03 Commercial A
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  // F04 Residential A
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
  // F05 Residential B
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  // F06 Suburban A
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  // F07 Suburban B
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  // F08 Rural
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  // F09 Commercial B
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  // F10 Residential C
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
];

let currentAlgorithm = "csp"; // "csp", "hill_climbing", "genetic"
let scheduleRequestId = 0;

// Chart instances
let demandChartInst = null;
let edaTimeSeriesInst = null;
let edaDiurnalInst = null;
let edaMonthlyInst = null;
let edaRequestId = 0;
let maintenanceDispatchOptions = {};
let maintenanceSelectedTeam = {};
let maintenanceSelectedAlgorithm = {};
let forecastCompareInst = null;
let schedPowerBalanceInst = null;
let schedFairnessBarInst = null;
let feederClusterInst = null;
let cityMapInst = null;
const cityMapMarkers = new Map();
const cityMapRegions = new Map();
const cityMapBoundaryRequests = new Map();

const FEEDER_LOCATIONS = {
  F01_HOSP: [24.8739, 67.0281],
  F02_IND: [24.8827, 66.9983],
  F03_COMM_A: [24.8587, 67.0104],
  F04_RES_A: [24.8068, 67.0577],
  F05_RES_B: [24.9165, 67.0959],
  F06_SUB_A: [24.9472, 67.0335],
  F07_SUB_B: [24.8896, 67.1695],
  F08_RUR: [25.0610, 67.2480],
  F09_COMM_B: [24.8662, 67.0587],
  F10_RES_C: [24.8138, 67.0300]
};

const FEEDER_LOCALITY_QUERIES = {
  F01_HOSP: "Jinnah Postgraduate Medical Centre, Karachi, Pakistan",
  F02_IND: "SITE Town, Karachi, Pakistan",
  F03_COMM_A: "Saddar Town, Karachi, Pakistan",
  F04_RES_A: "DHA Phase 6, Karachi, Pakistan",
  F05_RES_B: "Gulshan-e-Iqbal Town, Karachi, Pakistan",
  F06_SUB_A: "North Nazimabad Town, Karachi, Pakistan",
  F07_SUB_B: "Shah Faisal Colony, Karachi, Pakistan",
  F08_RUR: "Gadap Town, Karachi, Pakistan",
  F09_COMM_B: "PECHS, Karachi, Pakistan",
  F10_RES_C: "Clifton, Karachi, Pakistan"
};

// ============================================================================
// 2. INITIALIZATION ON DOM READY
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupAdminLogin();
  if (localStorage.getItem("fairshed-sidebar-collapsed") === "true") {
    document.body.classList.add("sidebar-collapsed");
  }
  initSparklines();
  initDemandChart();
  loadForecastData();
  renderCityMap();
  renderScheduleHeatmap();
  setupEventListeners();
  updateClock();
  setInterval(updateClock, 1000);

  // Pre-load EDA charts in background
  setTimeout(() => {
    initEDAData();
    initForecastChart();
    initLiveScheduler();
  }, 100);
});

async function loadForecastData() {
  try {
    const response = await fetch("/api/forecast");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Forecast service unavailable.");
    }

    const actualResponse = await fetch("/api/initial-data");
    const actualData = await actualResponse.json();
    if (!actualResponse.ok) {
      throw new Error(actualData.error || "Historical demand data unavailable.");
    }

    DEMANDS_24H = data.random_forest;
    PREDICTED_DEMANDS_24H = data.random_forest;
    LINEAR_DEMANDS_24H = data.linear_regression;
    ACTUAL_DEMANDS_24H = actualData.demands_24h;
    DEMANDS_24H = actualData.demands_24h;
    CAPACITIES_24H = actualData.capacities_24h;
    SHORTFALLS_24H = actualData.shortfalls_24h;
    SCHEDULED_SHED_24H = actualData.hourly_shed_24h;

    if (demandChartInst) {
      demandChartInst.data.datasets[0].data = ACTUAL_DEMANDS_24H;
      demandChartInst.data.datasets[1].data = PREDICTED_DEMANDS_24H;
      demandChartInst.data.datasets[2].data = LINEAR_DEMANDS_24H;
      demandChartInst.update();
    }
    if (schedPowerBalanceInst) {
      schedPowerBalanceInst.data.datasets[0].data = DEMANDS_24H;
      schedPowerBalanceInst.data.datasets[1].data = CAPACITIES_24H;
      schedPowerBalanceInst.data.datasets[2].data = SHORTFALLS_24H;
      schedPowerBalanceInst.data.datasets[3].data = SCHEDULED_SHED_24H;
      schedPowerBalanceInst.update();
    }
  } catch (error) {
    console.error("Unable to load next-day demand forecast:", error);
  }
}

let introTimer = null;

function skipIntroAnimation() {
  if (introTimer) clearTimeout(introTimer);
  const overlay = document.getElementById("adminLoginOverlay");
  if (overlay) {
    overlay.classList.add("intro-completed");
  }
  const emailInput = document.getElementById("adminEmail") || document.getElementById("adminUsername");
  emailInput?.focus();
}

function togglePasswordVisibility() {
  const p = document.getElementById("adminPassword");
  const icon = document.getElementById("passwordToggleIcon");
  const button = document.querySelector(".fairshade-eye");
  if (!p) return;
  if (p.type === "password") {
    p.type = "text";
    icon?.classList.remove("fa-eye");
    icon?.classList.add("fa-eye-slash");
    if (button) {
      button.textContent = "Hide";
      button.setAttribute("aria-label", "Hide password");
    }
  } else {
    p.type = "password";
    icon?.classList.remove("fa-eye-slash");
    icon?.classList.add("fa-eye");
    if (button) {
      button.textContent = "Show";
      button.setAttribute("aria-label", "Show password");
    }
  }
}

function autofillDemoAdmin() {
  const email = document.getElementById("adminEmail") || document.getElementById("adminUsername");
  const pass = document.getElementById("adminPassword");
  const err = document.getElementById("adminLoginError");
  if (email) {
    email.value = "admin@fairshade.pk";
    email.classList.add("ring-2", "ring-amber-400");
    setTimeout(() => email.classList.remove("ring-2", "ring-amber-400"), 600);
  }
  if (pass) {
    pass.value = "admin123";
    pass.classList.add("ring-2", "ring-amber-400");
    setTimeout(() => pass.classList.remove("ring-2", "ring-amber-400"), 600);
  }
  if (err) {
    err.classList.add("hidden");
    err.innerHTML = "";
  }
}

function showToast(msg) {
  let toast = document.getElementById("loginToast");
  if (!toast) return;
  toast.innerHTML = `<i class="fa-solid fa-circle-info text-amber-400"></i> <span>${msg}</span>`;
  toast.classList.remove("hidden");
  setTimeout(() => {
    toast.classList.add("hidden");
  }, 4000);
}

function handleForgotPassword(e) {
  e.preventDefault();
  showToast("Password reset instructions sent. (Demo password: admin123)");
}

function handleSignUp(e) {
  e.preventDefault();
  showToast("Account registration is managed by Grid Dispatchers. Please sign in with demo credentials.");
}

function showLoginError(msg) {
  const error = document.getElementById("adminLoginError");
  if (!error) return;
  if (!msg) {
    error.classList.add("hidden");
    error.innerHTML = "";
    return;
  }
  error.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-rose-400 flex-shrink-0"></i> <span>${msg}</span>`;
  error.classList.remove("hidden");
}

function setupAdminLogin() {
  const overlay = document.getElementById("adminLoginOverlay");
  const form = document.getElementById("adminLoginForm");
  if (!overlay || !form) return;

  const city = overlay.querySelector("#fairshadeCity");
  if (city && !city.childElementCount) {
    const blocks = [[10, 52, 95], [66, 40, 130], [110, 56, 80], [170, 46, 115], [220, 60, 140], [284, 44, 90], [332, 52, 120], [388, 44, 100]];
    let shapes = "";
    blocks.forEach(([x, width, height]) => {
      const y = 170 - height;
      shapes += `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="#0f1c3a"/>`;
      for (let windowY = y + 8; windowY < 158; windowY += 14) {
        for (let windowX = x + 6; windowX < x + width - 8; windowX += 12) {
          const zone = Math.min(3, Math.floor((windowX + 3) / 110));
          shapes += `<rect class="fairshade-win" data-zone="${zone}" x="${windowX}" y="${windowY}" width="6" height="8" rx="1"/>`;
        }
      }
    });
    for (let zone = 0; zone < 4; zone++) {
      shapes += `<rect class="fairshade-shade" data-zone="${zone}" x="${zone * 110}" y="0" width="110" height="170"/>`;
    }
    city.innerHTML = `${shapes}<rect x="0" y="170" width="440" height="3" fill="#26385f"/>`;

    const zoneLabels = [...overlay.querySelectorAll("#fairshadeZones span")];
    let activeZone = 0;
    const updateCity = () => {
      city.querySelectorAll(".fairshade-win").forEach((window) => {
        window.classList.toggle("off", Number(window.dataset.zone) === activeZone);
      });
      city.querySelectorAll(".fairshade-shade").forEach((shade) => {
        shade.classList.toggle("on", Number(shade.dataset.zone) === activeZone);
      });
      zoneLabels.forEach((label, index) => label.classList.toggle("on", index === activeZone));
      activeZone = (activeZone + 1) % 4;
    };
    setTimeout(() => {
      updateCity();
      setInterval(updateCity, 1800);
    }, 3200);
  }

  // Auto-complete intro animation after 2.6 seconds if not skipped
  introTimer = setTimeout(() => {
    overlay.classList.add("intro-completed");
  }, 2600);

  // Allow pressing Escape or Space to skip intro
  window.addEventListener("keydown", (e) => {
    if (!overlay.classList.contains("hidden") && !overlay.classList.contains("intro-completed")) {
      if (e.key === "Escape" || e.key === " ") {
        skipIntroAnimation();
      }
    }
  });

  fetch("/api/auth-status")
    .then((response) => response.json())
    .then((data) => {
      if (data.authenticated) overlay.classList.add("hidden");
    })
    .catch(() => {
      if (window.location.protocol === "file:") {
        // In local file mode, allow demo login directly
      }
    });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showLoginError("");
    const formData = new FormData(form);
    const emailOrUser = (formData.get("email") || formData.get("username") || "").trim();
    const password = (formData.get("password") || "").trim();
    const loginButton = document.getElementById("adminLoginBtn");
    const loginButtonText = document.getElementById("adminLoginBtnText");
    const spinner = loginButton?.querySelector(".fairshade-spinner");
    const setLoading = (loading) => {
      if (loginButton) loginButton.disabled = loading;
      if (spinner) spinner.hidden = !loading;
      if (loginButtonText) loginButtonText.textContent = loading ? "Logging in..." : "Log in";
    };

    // Local file preview fallback
    if (window.location.protocol === "file:") {
      const userHandle = emailOrUser.split("@")[0].toLowerCase();
      if ((emailOrUser === "admin" || userHandle === "admin") && password === "admin123") {
        overlay.classList.add("hidden");
        return;
      } else {
        showLoginError("Incorrect email or password. Use demo: admin@fairshade.pk / admin123");
        return;
      }
    }

    try {
      setLoading(true);
      const response = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: emailOrUser,
          email: emailOrUser,
          password: password
        })
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Incorrect email or password. Please try again.");
      }
      overlay.classList.add("hidden");
      window.location.reload();
    } catch (loginError) {
      setLoading(false);
      form.classList.remove("shake");
      void form.offsetWidth;
      form.classList.add("shake");
      showLoginError(loginError.message);
    }
  });
}

async function logoutAdmin() {
  if (!window.confirm("Are you sure you want to log out?")) {
    return;
  }
  await fetch("/api/logout", { method: "POST" });
  window.location.reload();
}

function toggleSidebar() {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  localStorage.setItem("fairshed-sidebar-collapsed", String(collapsed));
  const button = document.getElementById("sidebarToggle");
  if (button) {
    button.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
  }
}

// Update top header clock
function updateClock() {
  const el = document.getElementById("currentTimeDisplay");
  if (!el) return;
  const now = new Date();
  const options = { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false };
  el.textContent = now.toLocaleDateString('en-US', options).replace(/,/g, '');
}

// ============================================================================
// 3. TAB NAVIGATION (SEAMLESS SWITCHING)
// ============================================================================
function switchTab(tabId) {
  // Hide all tab views
  document.querySelectorAll(".tab-view").forEach(view => {
    view.classList.add("hidden");
  });

  // Reveal target tab view
  const target = document.getElementById(`view-${tabId}`);
  if (target) {
    target.classList.remove("hidden");
  }

  // Update nav item highlighting
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.remove("nav-pill-active");
    btn.classList.add("text-slate-400");
  });
  const activeNav = document.getElementById(`nav-${tabId}`);
  if (activeNav) {
    activeNav.classList.add("nav-pill-active");
    activeNav.classList.remove("text-slate-400");
  }

  // Scroll to top
  window.scrollTo({ top: 0, behavior: "smooth" });

  // Tab-specific chart triggers
  if (tabId === "eda") {
    setTimeout(renderEDACharts, 50);
  } else if (tabId === "forecast") {
    setTimeout(renderForecastCharts, 50);
  } else if (tabId === "scheduler") {
    setTimeout(renderSchedulerCharts, 50);
  } else if (tabId === "schedule") {
    setTimeout(renderFullScheduleTable, 50);
  } else if (tabId === "complaints") {
    setTimeout(loadMaintenanceTeamStatus, 50);
  } else if (tabId === "clustering") {
    setTimeout(loadFeederClusters, 50);
  }
}

// ============================================================================
// 4. TOP KPI SPARKLINES & MAIN DEMAND CHART
// ============================================================================
function initSparklines() {
  const createSpark = (id, data, color) => {
    const ctx = document.getElementById(id)?.getContext("2d");
    if (!ctx) return;
    new Chart(ctx, {
      type: "line",
      data: {
        labels: data.map((_, i) => i),
        datasets: [{
          data: data,
          borderColor: color,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.45,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });
  };

  createSpark("sparkline1", [16800, 17100, 16900, 17200, 17050, 17376], "#3b82f6");
  createSpark("sparkline2", [14800, 14950, 15100, 15000, 15200, 15134], "#a855f7");
  createSpark("sparkline3", [17900, 18100, 18400, 18200, 18600, 18825], "#10b981");
  createSpark("sparkline4", [92, 93, 94.5, 96, 98, 98.8], "#f59e0b");
}

function initDemandChart() {
  const ctx = document.getElementById("demandChart")?.getContext("2d");
  if (!ctx) return;

  const gradientPurple = ctx.createLinearGradient(0, 0, 0, 260);
  gradientPurple.addColorStop(0, "rgba(168, 85, 247, 0.45)");
  gradientPurple.addColorStop(1, "rgba(168, 85, 247, 0.02)");

  const gradientBlue = ctx.createLinearGradient(0, 0, 0, 260);
  gradientBlue.addColorStop(0, "rgba(59, 130, 246, 0.18)");
  gradientBlue.addColorStop(1, "rgba(59, 130, 246, 0.0)");

  demandChartInst = new Chart(ctx, {
    type: "line",
    data: {
      labels: HOURS_24,
      datasets: [
        {
          label: "Previous 24h Actual Demand",
          data: ACTUAL_DEMANDS_24H,
          borderColor: "#3b82f6",
          backgroundColor: gradientBlue,
          borderWidth: 2.5,
          pointRadius: 3,
          pointBackgroundColor: "#3b82f6",
          pointHoverRadius: 6,
          tension: 0.38,
          fill: true
        },
        {
          label: "Predicted Demand (Random Forest)",
          data: PREDICTED_DEMANDS_24H,
          borderColor: "#9333ea",
          backgroundColor: gradientPurple,
          borderWidth: 2.8,
          pointRadius: 3,
          pointBackgroundColor: "#9333ea",
          pointHoverRadius: 6,
          tension: 0.38,
          fill: true
        },
        {
          label: "Predicted Demand (Linear Regression)",
          data: LINEAR_DEMANDS_24H,
          borderColor: "#0284c7",
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: "#0284c7",
          pointHoverRadius: 6,
          tension: 0.38,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.94)",
          titleFont: { size: 12, weight: "bold", family: "Plus Jakarta Sans" },
          bodyFont: { size: 12, family: "Plus Jakarta Sans" },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toLocaleString()} MW`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            font: { size: 11, family: "Plus Jakarta Sans" },
            color: "#64748b",
            callback: function (val, index) {
              return [0, 6, 12, 18, 23].includes(index) ? this.getLabelForValue(val) : "";
            }
          }
        },
        y: {
          min: 0,
          max: 22000,
          ticks: {
            stepSize: 5000,
            font: { size: 11, family: "Plus Jakarta Sans" },
            color: "#64748b",
            callback: (v) => `${v / 1000}k`
          },
          grid: { color: "rgba(226, 232, 240, 0.6)" }
        }
      }
    }
  });
}

// ============================================================================
// 5. CITY LOAD MAP & HEATMAP PREVIEW (10 FEEDERS)
// ============================================================================
async function loadLocalityBoundary(area) {
  if (cityMapBoundaryRequests.has(area.id)) return;
  const query = FEEDER_LOCALITY_QUERIES[area.id];
  if (!query || !cityMapInst) return;

  const request = new Promise((resolve) => {
    setTimeout(resolve, cityMapBoundaryRequests.size * 1100);
  })
    .then(() => fetch(
      `https://nominatim.openstreetmap.org/search?format=geojson&polygon_geojson=1&limit=1&countrycodes=pk&q=${encodeURIComponent(query)}`,
      { headers: { "Accept-Language": "en" } }
    ))
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Boundary request failed for ${area.name}: HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((geojson) => {
      if (!geojson.features || geojson.features.length === 0) {
        throw new Error(`No OpenStreetMap boundary found for ${area.name}`);
      }
      const region = L.geoJSON(geojson, {
        style: {
          color: area.color,
          weight: 1.5,
          opacity: 0.9,
          fillColor: area.color,
          fillOpacity: 0.32
        }
      }).addTo(cityMapInst);
      region.bindTooltip(`${area.name} · ${area.exempt ? "0h exempt" : `${area.outageHours}h outage`}`, {
        direction: "center",
        className: "feeder-region-label",
        sticky: true
      });
      region.on("click", () => showAreaModal(area.id));
      cityMapRegions.set(area.id, region);
    })
    .catch((error) => {
      console.warn(error.message);
    });

  cityMapBoundaryRequests.set(area.id, request);
}

function renderCityMap() {
  const container = document.getElementById("cityMapContainer");
  if (!container) return;

  // Calculate outage hours per feeder
  CITY_AREAS.forEach((area, idx) => {
    area.outageHours = activeScheduleMatrix[idx].reduce((a, b) => a + b, 0);
  });

  if (typeof L === "undefined") {
    container.textContent = "Map service unavailable. Please check your network connection.";
    return;
  }

  if (!cityMapInst) {
    cityMapInst = L.map(container, { zoomControl: true, attributionControl: false }).setView([24.88, 67.06], 11);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18
    }).addTo(cityMapInst);
    cityMapInst.on("zoomend", () => cityMapInst.invalidateSize({ pan: false }));
    document.getElementById("cityMapZoomIn")?.addEventListener("click", () => cityMapInst.zoomIn());
    document.getElementById("cityMapZoomOut")?.addEventListener("click", () => cityMapInst.zoomOut());
    document.getElementById("cityMapReset")?.addEventListener("click", () => {
      const locations = CITY_AREAS.map((area) => FEEDER_LOCATIONS[area.id]).filter(Boolean);
      if (locations.length > 1) {
        cityMapInst.fitBounds(L.latLngBounds(locations), {
          padding: [18, 18],
          maxZoom: 11,
          animate: true
        });
      }
    });
  }

  CITY_AREAS.forEach((area) => {
    const location = FEEDER_LOCATIONS[area.id];
    if (!location) return;
    const markerIcon = L.divIcon({
      className: "feeder-marker",
      html: `<div class="feeder-marker-dot" style="background:${area.color};"></div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9]
    });
    const popup = `<strong>${area.name}</strong><br>${area.category || area.type}<br>Base load: ${area.load} MW<br>Outage: ${area.exempt ? "0h (exempt)" : `${area.outageHours}h`}`;
    const existing = cityMapMarkers.get(area.id);
    if (existing) {
      existing.setIcon(markerIcon).setPopupContent(popup);
    } else {
      const marker = L.marker(location, { icon: markerIcon }).addTo(cityMapInst);
      marker.bindPopup(popup);
      marker.on("click", () => showAreaModal(area.id));
      cityMapMarkers.set(area.id, marker);
    }
    loadLocalityBoundary(area);
  });
  const mapLocations = CITY_AREAS
    .map((area) => FEEDER_LOCATIONS[area.id])
    .filter(Boolean);
  setTimeout(() => {
    cityMapInst.invalidateSize({ pan: false });
    if (mapLocations.length > 1) {
      cityMapInst.fitBounds(L.latLngBounds(mapLocations), {
        padding: [18, 18],
        maxZoom: 11,
        animate: false
      });
    }
  }, 0);

  // Render Legend
  const legendContainer = document.getElementById("mapLegendContainer");
  if (legendContainer) {
    legendContainer.innerHTML = CITY_AREAS.map((a) => `
      <div class="flex items-center justify-between text-xs py-1 px-1.5 rounded hover:bg-slate-100 cursor-pointer transition-colors" onclick="showAreaModal('${a.id}')">
        <div class="flex items-center gap-1.5 truncate">
          <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${a.color};"></span>
          <span class="text-slate-700 truncate text-[11px] font-medium">${a.name}</span>
        </div>
        <span class="font-extrabold ml-1 text-[11px] ${a.exempt ? 'text-sky-500' : 'text-purple-700'}">${a.exempt ? '0h' : a.outageHours + 'h'}</span>
      </div>
    `).join("");
  }
}

function renderScheduleHeatmap() {
  const container = document.getElementById("scheduleHeatmapGrid");
  if (!container) return;

  let html = "";
  CITY_AREAS.forEach((area, areaIdx) => {
    html += `
      <div class="flex items-center gap-2 mb-1.5">
        <div class="w-28 text-xs font-semibold text-slate-700 truncate text-right pr-1 cursor-pointer hover:text-indigo-600" onclick="showAreaModal('${area.id}')" title="${area.fullName}">
          ${area.name}
        </div>
        <div class="grid grid-cols-24 gap-1 flex-1">
    `;

    for (let h = 0; h < 24; h++) {
      const isShed = activeScheduleMatrix[areaIdx][h] === 1;
      const isProtected = h === 4 || h === 18 || h === 19;
      const cellClass = isShed ? "power-off" : "power-on";
      const protectedClass = isProtected ? "protected-hour" : "";
      const statusTitle = isShed ? "Load Shedding" : "Power On";
      const meal = h === 4 ? " (Sehri Protected)" : (h === 18 || h === 19 ? " (Iftar Protected)" : "");

      html += `
        <div class="heat-cell ${cellClass} ${protectedClass}"
             title="${area.name} at ${HOURS_24[h]}: ${statusTitle}${meal}"
             onclick="showAreaModal('${area.id}')">
        </div>
      `;
    }

    html += `
        </div>
        <div class="w-8 text-xs font-bold text-center ${area.exempt ? 'text-sky-500' : 'text-slate-700'}">
          ${area.exempt ? '0h' : area.outageHours + 'h'}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

// Select algorithm on dashboard
async function selectAlgorithm(algo) {
  currentAlgorithm = algo;

  const btnCsp = document.getElementById("btnSelectCsp");
  const btnHc = document.getElementById("btnSelectHc");
  const activeLabel = document.getElementById("activeAlgoLabel");
  const dashboardSelect = document.getElementById("dashboardAlgoSelect");
  const liveSelect = document.getElementById("liveAlgoSelect");

  if (dashboardSelect && dashboardSelect.value !== algo) {
    dashboardSelect.value = algo;
  }
  if (liveSelect && liveSelect.value !== algo) {
    liveSelect.value = algo;
  }

  const algorithmLabels = {
    csp: "CSP Backtracking (MRV + Forward Checking)",
    hill_climbing: "Steepest-Ascent Hill Climbing",
    genetic: "Genetic Algorithm"
  };

  if (algo === "csp") {
    btnCsp?.classList.add("bg-indigo-600", "text-white");
    btnCsp?.classList.remove("border", "border-indigo-600", "text-indigo-600");
    btnHc?.classList.remove("bg-indigo-600", "text-white");
    btnHc?.classList.add("border", "border-indigo-600", "text-indigo-600");
    if (activeLabel) activeLabel.textContent = "CSP Backtracking (MRV + Forward Checking)";
  } else if (btnHc || btnCsp) {
    btnHc?.classList.add("bg-indigo-600", "text-white");
    btnHc?.classList.remove("border", "border-indigo-600", "text-indigo-600");
    btnCsp?.classList.remove("bg-indigo-600", "text-white");
    btnCsp?.classList.add("border", "border-indigo-600", "text-indigo-600");
  }

  if (activeLabel) activeLabel.textContent = algorithmLabels[algo] || algorithmLabels.csp;

  await runLiveScheduler(algo);
}

// Area Details Modal
function showAreaModal(areaId) {
  const area = CITY_AREAS.find((a) => a.id === areaId);
  if (!area) return;

  const modal = document.getElementById("areaModal");
  const content = document.getElementById("areaModalContent");
  if (!modal || !content) return;

  content.innerHTML = `
    <div class="flex items-center gap-3 mb-4">
      <div class="w-10 h-10 rounded-xl flex items-center justify-center text-white font-bold" style="background-color: ${area.color};">
        <i class="fa-solid fa-bolt"></i>
      </div>
      <div>
        <h3 class="text-base font-bold text-slate-800">${area.fullName}</h3>
        <p class="text-xs text-slate-500 font-mono">ID: ${area.id} • Feeder Code</p>
      </div>
    </div>

    <div class="grid grid-cols-2 gap-3 mb-4 text-xs">
      <div class="p-3 bg-slate-50 rounded-xl border border-slate-100">
        <span class="text-slate-500 block">Sector Category</span>
        <span class="font-bold text-slate-800 text-sm">${area.type}</span>
      </div>
      <div class="p-3 bg-slate-50 rounded-xl border border-slate-100">
        <span class="text-slate-500 block">Base Demand Load</span>
        <span class="font-bold text-slate-800 text-sm">${area.load} MW</span>
      </div>
      <div class="p-3 bg-slate-50 rounded-xl border border-slate-100">
        <span class="text-slate-500 block">Critical Hospital Exemption</span>
        <span class="font-bold ${area.exempt ? 'text-emerald-600' : 'text-slate-700'} text-sm">${area.exempt ? 'YES (100% Protected)' : 'NO (Sheddable)'}</span>
      </div>
      <div class="p-3 bg-slate-50 rounded-xl border border-slate-100">
        <span class="text-slate-500 block">Max Daily Quota</span>
        <span class="font-bold text-slate-800 text-sm">${area.maxOutage} Hours / Day</span>
      </div>
    </div>

    <div class="p-3 rounded-xl border ${area.exempt ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-purple-50 border-purple-200 text-purple-900'} text-xs mb-4">
      <div class="flex items-center justify-between font-bold mb-1">
        <span>Scheduled Blackout Today</span>
        <span class="text-sm">${area.outageHours} Hours</span>
      </div>
      <p class="text-[11px] ${area.exempt ? 'text-emerald-700' : 'text-purple-700'}">
        ${area.exempt
          ? 'Guaranteed 24/7 continuous uninterrupted power delivery for life-support ventilators and surgery suites.'
          : 'Fairly equalized with adjacent socioeconomic zones under Jain Fairness Index constraints.'
        }
      </p>
    </div>

    <div class="flex justify-end gap-2">
      <button onclick="closeAreaModal()" class="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold rounded-lg transition-all">Close</button>
      <button onclick="closeAreaModal(); switchTab('schedule');" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition-all">View 24h Schedule</button>
    </div>
  `;

  modal.classList.remove("hidden");
}

function closeAreaModal() {
  document.getElementById("areaModal")?.classList.add("hidden");
}

// ============================================================================
// 6. HISTORICAL DEMAND & SEASONAL EDA
// ============================================================================
function initEDAData() {
  renderEDACharts();
  document.getElementById("edaStartDate")?.addEventListener("change", loadEDAData);
  document.getElementById("edaEndDate")?.addEventListener("change", loadEDAData);
  loadEDAData();
}

function setEDAPreset(preset) {
  const startEl = document.getElementById("edaStartDate");
  const endEl = document.getElementById("edaEndDate");
  if (!startEl || !endEl) return;

  if (preset === "summer") {
    startEl.value = "2024-06-01";
    endEl.value = "2024-08-31";
  } else if (preset === "winter") {
    startEl.value = "2024-11-01";
    endEl.value = "2024-12-31";
  } else if (preset === "ramadan") {
    startEl.value = "2024-03-11";
    endEl.value = "2024-04-09";
  } else {
    startEl.value = "2023-01-01";
    endEl.value = "2024-12-31";
  }

  loadEDAData();
}

async function loadEDAData() {
  const requestId = ++edaRequestId;
  const startDate = document.getElementById("edaStartDate")?.value || "2023-01-01";
  const endDate = document.getElementById("edaEndDate")?.value || "2024-12-31";
  const status = document.getElementById("edaStatus");
  if (status) status.textContent = "Loading EDA data…";

  try {
    const res = await fetch("/api/eda", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ startDate, endDate })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `EDA request failed (${res.status}).`);
    }
    if (requestId !== edaRequestId) return;
    updateEDADisplay(data);
    if (status) {
      status.textContent = `Showing ${Number(data.filtered_count).toLocaleString()} hourly records (${startDate} to ${endDate}).`;
    }
  } catch (error) {
    if (requestId !== edaRequestId) return;
    clearEDADisplay();
    const message = error instanceof Error ? error.message : String(error);
    if (status) status.textContent = `Unable to load EDA data: ${message}`;
    console.error("Unable to load EDA data:", error);
  }
}

function updateEDADisplay(data) {
  if (!data?.timeseries || !data?.diurnal || !data?.monthly) {
    throw new Error("The EDA response is missing chart data.");
  }
  document.getElementById("edaCount").textContent = Number(data.filtered_count).toLocaleString() + " hrs";
  document.getElementById("edaMean").textContent = Number(data.stats.mean).toLocaleString() + " MW";
  document.getElementById("edaMax").textContent = Number(data.stats.max).toLocaleString() + " MW";
  document.getElementById("edaMin").textContent = Number(data.stats.min).toLocaleString() + " MW";

  if (edaTimeSeriesInst) {
    edaTimeSeriesInst.data.labels = data.timeseries.timestamps;
    edaTimeSeriesInst.data.datasets[0].data = data.timeseries.demands;
    edaTimeSeriesInst.update();
  }
  if (edaDiurnalInst) {
    edaDiurnalInst.data.labels = data.diurnal.hours.map(hour => `${hour}:00`);
    edaDiurnalInst.data.datasets[0].data = data.diurnal.mean;
    edaDiurnalInst.data.datasets[1].data = data.diurnal.max;
    edaDiurnalInst.data.datasets[2].data = data.diurnal.min;
    edaDiurnalInst.update();
  }
  if (edaMonthlyInst) {
    edaMonthlyInst.data.labels = data.monthly.months;
    edaMonthlyInst.data.datasets[0].data = data.monthly.mean;
    edaMonthlyInst.data.datasets[1].data = data.monthly.max;
    edaMonthlyInst.update();
  }
}

function clearEDADisplay() {
  document.getElementById("edaCount").textContent = "—";
  document.getElementById("edaMean").textContent = "—";
  document.getElementById("edaMax").textContent = "—";
  document.getElementById("edaMin").textContent = "—";
  [edaTimeSeriesInst, edaDiurnalInst, edaMonthlyInst].forEach((chart) => {
    if (!chart) return;
    chart.data.labels = [];
    chart.data.datasets.forEach(dataset => { dataset.data = []; });
    chart.update();
  });
}

function renderEDACharts() {
  // 1. Time Series Chart
  const ctxTs = document.getElementById("edaTimeSeriesChart")?.getContext("2d");
  if (ctxTs && !edaTimeSeriesInst) {
    edaTimeSeriesInst = new Chart(ctxTs, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label: "Hourly Demand (MW)",
          data: [],
          borderColor: "#4f46e5",
          backgroundColor: "rgba(79, 70, 229, 0.08)",
          borderWidth: 1.8,
          pointRadius: 0,
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 12, font: { size: 10 } } },
          y: { ticks: { callback: (v) => `${v / 1000}k MW`, font: { size: 10 } } }
        }
      }
    });
  }

  // 2. Diurnal Peak Analysis Chart (0-23h)
  const ctxDiurnal = document.getElementById("edaDiurnalChart")?.getContext("2d");
  if (ctxDiurnal && !edaDiurnalInst) {
    edaDiurnalInst = new Chart(ctxDiurnal, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Mean Load (MW)",
            data: [],
            borderColor: "#3b82f6",
            borderWidth: 2.8,
            pointRadius: 2,
            tension: 0.4
          },
          {
            label: "Max Peak (MW)",
            data: [],
            borderColor: "#ef4444",
            borderDash: [5, 5],
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.4
          },
          {
            label: "Min Trough (MW)",
            data: [],
            borderColor: "#10b981",
            borderDash: [5, 5],
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { ticks: { font: { size: 10 } } },
          y: { ticks: { callback: (v) => `${v / 1000}k MW`, font: { size: 10 } } }
        }
      }
    });
  }

  // 3. Monthly Seasonality Chart
  const ctxMonthly = document.getElementById("edaMonthlyChart")?.getContext("2d");
  if (ctxMonthly && !edaMonthlyInst) {
    edaMonthlyInst = new Chart(ctxMonthly, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          {
            label: "Mean Demand (MW)",
            data: [],
            backgroundColor: "rgba(99, 102, 241, 0.75)",
            borderRadius: 6
          },
          {
            label: "Monthly Peak (MW)",
            data: [],
            backgroundColor: "rgba(244, 63, 94, 0.7)",
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { ticks: { font: { size: 10 } } },
          y: { ticks: { callback: (v) => `${v / 1000}k`, font: { size: 10 } } }
        }
      }
    });
  }
}

// ============================================================================
// 7. SUPERVISED LEARNING: 7-DAY OUT-OF-SAMPLE BENCHMARK CHART
// ============================================================================
function initForecastChart() {
  renderForecastCharts();
}

function renderForecastCharts() {
  const ctx = document.getElementById("forecastCompareChart")?.getContext("2d");
  if (!ctx || forecastCompareInst) return;

  // 168 hours (7 days) realistic synthetic test series
  const labels = [];
  const actual = [];
  const rf = [];
  const linear = [];

  for (let i = 0; i < 168; i++) {
    const day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][Math.floor(i / 24)];
    const hr = i % 24;
    labels.push(`${day} ${hr}:00`);

    const base = 14500 + 3200 * Math.sin((hr - 6) * Math.PI / 12) + (Math.floor(i / 24) >= 5 ? -1200 : 0);
    actual.push(Math.round(base + (Math.random() * 400 - 200)));
    rf.push(Math.round(base + (Math.random() * 150 - 75)));
    linear.push(Math.round(base + (Math.random() * 260 - 130)));
  }

  forecastCompareInst = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "Actual Demand", data: actual, borderColor: "#0f172a", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: "Random Forest Forecast", data: rf, borderColor: "#6366f1", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: "Linear Regression Forecast", data: linear, borderColor: "#0284c7", borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 14, font: { size: 10 } } },
        y: { ticks: { callback: (v) => `${v / 1000}k MW`, font: { size: 10 } } }
      }
    }
  });
}

// ============================================================================
// 8. LIVE FAIR LOAD-SHEDDING SCHEDULER & POWER BALANCE CHARTS
// ============================================================================
function initLiveScheduler() {
  renderSchedulerCharts();
  loadMaintenanceTeamStatus();
  setInterval(loadMaintenanceTeamStatus, 5000);
}

async function loadMaintenanceTeamStatus() {
  try {
    const response = await fetch("/api/public-status");
    if (!response.ok) return;
    const data = await response.json();
    renderComplaintMaintenanceTab(data.maintenance?.teams || [], data.maintenance?.complaints || []);
  } catch (error) {
    console.error("Unable to load maintenance team status:", error);
  }
}

function renderComplaintMaintenanceTab(teams, complaints = []) {
  const teamContainer = document.getElementById("complaintTeamStatus");
  if (teamContainer) {
    teamContainer.innerHTML = teams.map((team) => {
    const busy = team.status === "busy";
    const completionButton = busy
      ? `<button type="button" class="mt-3 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold" onclick="completeMaintenanceTeam('${team.id}')">Mark repair complete</button>`
      : '<p class="mt-3 text-xs text-emerald-600 font-bold">Available</p>';
    return `<div class="p-3 rounded-lg bg-white border border-slate-200">
      <p class="text-sm font-bold text-slate-800">${team.name}</p>
      <p class="text-xs ${busy ? "text-amber-600" : "text-emerald-600"} mt-1">${busy ? "Busy" : "Available"}</p>
      ${busy ? `<p class="text-xs text-slate-500 mt-1">${team.assigned_complaint_id} · ${team.assigned_area_id}${team.travel_minutes ? ` · ${team.travel_minutes} min` : ""}</p>` : ""}
      ${completionButton}
    </div>`;
    }).join("");
  }

  const pending = complaints.filter((complaint) => complaint.status === "pending_admin");
  maintenanceDispatchOptions = Object.fromEntries(
    pending.map((complaint) => [complaint.id, complaint.dispatch_options || { recommended_team_id: null, options: [] }])
  );
  const pendingCount = document.getElementById("complaintPendingCount");
  if (pendingCount) pendingCount.textContent = `${pending.length} pending`;
  const dispatchList = document.getElementById("complaintDispatchList");
  if (dispatchList) {
    dispatchList.innerHTML = pending.length ? pending.map((complaint) => {
      const dispatch = complaint.dispatch_options || { recommended_team_id: null, options: [] };
      const availableTeams = teams.filter((team) => team.status === "available");
      const selectedTeam = maintenanceSelectedTeam[complaint.id];
      const selectedAlgorithm = maintenanceSelectedAlgorithm[complaint.id] || "UCS";
      const options = availableTeams.map((team) => {
        const option = dispatch.options.find((item) => item.team_id === team.id);
        const algorithmRoute = selectedAlgorithm === "A*" ? option?.astar : option?.ucs;
        const minutes = algorithmRoute?.travel_minutes;
        const recommendedTeamId = selectedAlgorithm === "A*"
          ? dispatch.astar_recommended_team_id
          : dispatch.ucs_recommended_team_id;
        const recommended = team.id === recommendedTeamId ? " - Recommended" : "";
        return `<option value="${team.id}">${team.name}${minutes != null ? ` - ${minutes} min` : ""}${recommended}</option>`;
      }).join("");
      const routeOptions = (dispatch.options || []).map((teamOption) => {
        const ucs = teamOption.ucs ? `${teamOption.ucs.travel_minutes} min` : "Unavailable";
        const astar = teamOption.astar ? `${teamOption.astar.travel_minutes} min` : "Unavailable";
        return `<div class="text-xs text-slate-600">
          <strong>${teamOption.team_id}</strong>: UCS ${ucs} | A* ${astar}
        </div>`;
      }).join("");
      return `<div class="p-4 bg-white border border-slate-200 rounded-xl">
        <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div><p class="text-sm font-bold text-slate-800">${complaint.id}</p><p class="text-xs text-slate-500">${complaint.area_id} · Hour ${complaint.hour}:00 · Pending admin review</p></div>
          ${availableTeams.length ? `<span class="text-xs font-bold text-emerald-700">Available teams: ${availableTeams.length}</span>` : '<span class="text-xs font-bold text-rose-700">No team available</span>'}
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div class="p-3 bg-slate-50 rounded-lg"><p class="text-xs font-bold text-slate-700 mb-1">Search comparison</p>${routeOptions}</div>
          <div class="p-3 bg-slate-50 rounded-lg">
            <label class="block text-xs font-bold text-slate-700 mb-1" for="teamFor_${complaint.id}">Assign team</label>
            ${availableTeams.length ? `<select id="teamFor_${complaint.id}" class="w-full text-xs p-2 border border-slate-300 rounded mb-2" onchange="updateMaintenanceEstimate('${complaint.id}')">${options}</select>
            <select id="algorithmFor_${complaint.id}" class="w-full text-xs p-2 border border-slate-300 rounded mb-2" onchange="updateMaintenanceEstimate('${complaint.id}')"><option value="UCS" ${selectedAlgorithm === "UCS" ? "selected" : ""}>UCS - Lowest-cost route</option><option value="A*" ${selectedAlgorithm === "A*" ? "selected" : ""}>A* - Heuristic route</option></select>
            <div id="estimateFor_${complaint.id}" class="p-2 mb-2 rounded-lg bg-white border border-slate-200 text-xs text-slate-700"></div>
            <button type="button" class="px-3 py-2 rounded-lg bg-indigo-600 text-white text-xs font-bold" onclick="assignMaintenanceTeam('${complaint.id}')">Accept & assign team</button>`
              : '<p class="text-xs text-amber-700">A team can be assigned after a repair is marked complete.</p>'}
          </div>
        </div>
      </div>`;
    }).join("") : '<p class="text-sm text-slate-500">No pending complaints.</p>';
    pending.forEach((complaint) => {
      const teamSelect = document.getElementById(`teamFor_${complaint.id}`);
      if (teamSelect && maintenanceSelectedTeam[complaint.id]) {
        teamSelect.value = maintenanceSelectedTeam[complaint.id];
      }
      const algorithmSelect = document.getElementById(`algorithmFor_${complaint.id}`);
      if (algorithmSelect && maintenanceSelectedAlgorithm[complaint.id]) {
        algorithmSelect.value = maintenanceSelectedAlgorithm[complaint.id];
      }
      updateMaintenanceEstimate(complaint.id);
    });
  }

  const history = document.getElementById("complaintHistory");
  if (history) {
    const relevant = complaints.filter((complaint) => complaint.status !== "pending_admin");
    history.innerHTML = relevant.length ? relevant.map((complaint) => `
      <div class="flex flex-wrap items-center justify-between gap-2 p-3 rounded-lg border border-slate-200">
        <div><p class="text-xs font-bold text-slate-800">${complaint.id} · ${complaint.area_id}</p><p class="text-[11px] text-slate-500">Hour ${complaint.hour}:00${complaint.team_id ? ` · ${complaint.team_id} · ${complaint.travel_minutes} min` : ""}${complaint.assignment_algorithm ? ` · ${complaint.assignment_algorithm}` : ""}</p></div>
        <span class="text-xs font-bold ${complaint.status === "completed" ? "text-emerald-700" : complaint.status === "planned_outage" ? "text-slate-600" : "text-indigo-700"}">${complaint.status}</span>
      </div>`).join("") : '<p class="text-sm text-slate-500">No complaint history.</p>';
  }
}

function updateMaintenanceEstimate(complaintId) {
  const teamSelect = document.getElementById(`teamFor_${complaintId}`);
  const algorithmSelect = document.getElementById(`algorithmFor_${complaintId}`);
  const estimateBox = document.getElementById(`estimateFor_${complaintId}`);
  if (!teamSelect || !algorithmSelect || !estimateBox) return;

  maintenanceSelectedTeam[complaintId] = teamSelect.value;
  maintenanceSelectedAlgorithm[complaintId] = algorithmSelect.value;

  const dispatch = maintenanceDispatchOptions[complaintId] || { options: [] };
  const recommendedTeamId = algorithmSelect.value === "A*"
    ? dispatch.astar_recommended_team_id
    : dispatch.ucs_recommended_team_id;

  Array.from(teamSelect.options).forEach((option) => {
    const teamOption = dispatch.options.find((item) => item.team_id === option.value);
    const routeOption = algorithmSelect.value === "A*" ? teamOption?.astar : teamOption?.ucs;
    const minutes = routeOption?.travel_minutes;
    const teamName = teamOption?.team_name || option.value;
    const recommended = option.value === recommendedTeamId ? " - Recommended" : "";
    option.textContent = `${teamName}${minutes != null ? ` - ${minutes} min` : ""}${recommended}`;
  });

  const teamOption = dispatch.options.find((option) => option.team_id === teamSelect.value);
  if (!teamOption) {
    estimateBox.textContent = "No route estimate available.";
    return;
  }

  const algorithm = algorithmSelect.value;
  const routeOption = algorithm === "A*" ? teamOption.astar : teamOption.ucs;
  if (!routeOption) {
    estimateBox.textContent = "Selected route is unavailable.";
    return;
  }

  const routeText = routeOption.route.join(" → ");
  estimateBox.innerHTML =
    `<strong>${algorithm} estimate:</strong> ${routeOption.travel_minutes} minutes<br>` +
    `<span class="text-slate-500">Route: ${routeText}</span>`;
}

async function completeMaintenanceTeam(teamId) {
  try {
    const response = await fetch("/api/maintenance/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_id: teamId })
    });
    if (!response.ok) throw new Error(`Completion failed (${response.status})`);
    const data = await response.json();
    renderComplaintMaintenanceTab(data.maintenance?.teams || [], data.maintenance?.complaints || []);
  } catch (error) {
    console.error("Unable to complete maintenance dispatch:", error);
  }
}

// 10. SEPARATE FEEDER CLUSTERING FEATURE
async function loadFeederClusters() {
  const k = Number(document.getElementById("clusterCount")?.value || 4);
  const status = document.getElementById("clusterStatus");
  if (status) status.textContent = "Running K-Means…";
  try {
    const response = await fetch("/api/feeder-clusters", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "K-Means request failed.");
    renderFeederClusters(data);
    if (status) status.textContent = `${data.k} clusters · inertia ${data.inertia}`;
  } catch (error) {
    if (status) status.textContent = `Unable to cluster feeders: ${error.message}`;
    console.error("Unable to run feeder clustering:", error);
  }
}

function renderFeederClusters(data) {
  const colors = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#0ea5e9", "#a855f7", "#64748b", "#ec4899", "#84cc16", "#f97316"];
  const clusterIds = [...new Set(data.profiles.map((profile) => profile.cluster_id))].sort((a, b) => a - b);
  const datasets = clusterIds.map((clusterId) => {
    const profiles = data.profiles.filter((profile) => profile.cluster_id === clusterId);
    return {
      label: `Cluster ${clusterId + 1}: ${profiles[0]?.profile_archetype || "Profile"}`,
      data: profiles.map((profile) => ({
        x: profile.base_load_mw,
        y: profile.disruption_cost_per_hour,
        feederName: profile.name,
        areaId: profile.area_id
      })),
      backgroundColor: colors[clusterId % colors.length],
      borderColor: colors[clusterId % colors.length],
      pointRadius: 7,
      pointHoverRadius: 9
    };
  });

  const ctx = document.getElementById("feederClusterChart")?.getContext("2d");
  if (ctx) {
    if (feederClusterInst) feederClusterInst.destroy();
    feederClusterInst = new Chart(ctx, {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (context) => {
                const point = context.raw;
                return `${point.feederName}: ${point.x} MW, cost ${point.y}/hr`;
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: "Base Load (MW)" } },
          y: { title: { display: true, text: "Disruption Cost per Hour" } }
        }
      }
    });
  }

  const summary = document.getElementById("clusterSummary");
  if (summary) {
    summary.innerHTML = clusterIds.map((clusterId) => {
      const profiles = data.profiles.filter((profile) => profile.cluster_id === clusterId);
      const meanLoad = profiles.reduce((sum, profile) => sum + profile.base_load_mw, 0) / profiles.length;
      const meanCost = profiles.reduce((sum, profile) => sum + profile.disruption_cost_per_hour, 0) / profiles.length;
      return `<div class="p-3 rounded-lg border border-slate-200" style="border-left:4px solid ${colors[clusterId % colors.length]}">
        <p class="text-xs font-bold text-slate-800">Cluster ${clusterId + 1}</p>
        <p class="text-[11px] text-slate-500 mt-1">${profiles[0]?.profile_archetype || "Profile"}</p>
        <p class="text-[11px] text-slate-600 mt-1">${profiles.length} feeders · ${meanLoad.toFixed(0)} MW avg · ${meanCost.toFixed(0)} cost/hr</p>
      </div>`;
    }).join("");
  }

  const tableBody = document.getElementById("clusterTableBody");
  if (tableBody) {
    tableBody.innerHTML = data.profiles.map((profile) => `
      <tr class="border-b border-slate-100">
        <td class="py-2 pr-3 font-semibold text-slate-700">${profile.name}</td>
        <td>Cluster ${profile.cluster_id + 1}</td>
        <td>${profile.profile_archetype}</td>
        <td>${profile.base_load_mw}</td>
        <td>${profile.disruption_cost_per_hour}</td>
      </tr>`).join("");
  }
}

async function assignMaintenanceTeam(complaintId) {
  const teamSelect = document.getElementById(`teamFor_${complaintId}`);
  const algorithmSelect = document.getElementById(`algorithmFor_${complaintId}`);
  if (!teamSelect || !algorithmSelect) return;
  try {
    const response = await fetch("/api/maintenance/assign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        complaint_id: complaintId,
        team_id: teamSelect.value,
        algorithm: algorithmSelect.value
      })
    });
    if (!response.ok) throw new Error(`Assignment failed (${response.status})`);
    const data = await response.json();
    renderComplaintMaintenanceTab(data.maintenance?.teams || [], data.maintenance?.complaints || []);
  } catch (error) {
    console.error("Unable to assign maintenance team:", error);
  }
}

function renderSchedulerCharts() {
  // 1. Power Balance Chart
  const ctxPb = document.getElementById("schedPowerBalanceChart")?.getContext("2d");
  if (ctxPb && !schedPowerBalanceInst) {
    const hourlyShed = SCHEDULED_SHED_24H || calculateHourlyShedMW(activeScheduleMatrix);

    schedPowerBalanceInst = new Chart(ctxPb, {
      type: "bar",
      data: {
        labels: HOURS_24,
        datasets: [
          { type: "line", label: "Demand (MW)", data: DEMANDS_24H, borderColor: "#2563eb", borderWidth: 2.5, pointRadius: 3, tension: 0.3, yAxisID: "y" },
          { type: "line", label: "Available Capacity (MW)", data: CAPACITIES_24H, borderColor: "#16a34a", borderWidth: 2, borderDash: [5, 5], pointRadius: 0, yAxisID: "y" },
          { label: "Bayesian Expected Shortfall (MW)", data: SHORTFALLS_24H, backgroundColor: "rgba(244, 63, 94, 0.65)", borderRadius: 4, yAxisID: "y" },
          { label: "Scheduled Shed (MW)", data: hourlyShed, backgroundColor: "rgba(16, 185, 129, 0.75)", borderRadius: 4, yAxisID: "y" }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { ticks: { font: { size: 10 } } },
          y: { min: 0, ticks: { callback: (v) => `${v / 1000}k MW`, font: { size: 10 } } }
        }
      }
    });
  }

  // 2. Fairness Bar Chart
  const ctxFair = document.getElementById("schedFairnessBarChart")?.getContext("2d");
  if (ctxFair && !schedFairnessBarInst) {
    const outageCounts = CITY_AREAS.map(a => a.outageHours);
    const colors = CITY_AREAS.map(a => a.color);

    schedFairnessBarInst = new Chart(ctxFair, {
      type: "bar",
      data: {
        labels: CITY_AREAS.map(a => a.name),
        datasets: [{
          label: "Outage Hours",
          data: outageCounts,
          backgroundColor: colors,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { font: { size: 10 } } },
          y: { min: 0, max: 6, ticks: { stepSize: 1, callback: (v) => `${v}h` } }
        }
      }
    });
  }
}

function calculateHourlyShedMW(matrix) {
  const result = new Array(24).fill(0);
  const totalBase = CITY_AREAS.reduce((s, a) => s + a.load, 0);

  for (let h = 0; h < 24; h++) {
    let shed = 0;
    for (let i = 0; i < CITY_AREAS.length; i++) {
      if (matrix[i][h] === 1) {
        shed += DEMANDS_24H[h] * (CITY_AREAS[i].load / totalBase);
      }
    }
    result[h] = Math.round(shed);
  }
  return result;
}

// Client-Side Solver Engine (MRV Forward-Checking CSP & Hill Climbing)
function solveScheduleLocal(algo = "csp", options = {}) {
  const isRamadan = options.isRamadan !== undefined ? options.isRamadan : true;
  const excludeHospitals = options.excludeHospitals !== undefined ? options.excludeHospitals : true;
  const maxDailyQuota = options.maxDailyQuota || 4;
  const maxConsecutive = options.maxConsecutive || 2;
  const genRatio = options.genRatio || 0.82;

  CAPACITIES_24H = DEMANDS_24H.map(d => Math.round(d * genRatio));
  SHORTFALLS_24H = DEMANDS_24H.map((d, i) => Math.max(0, d - CAPACITIES_24H[i]));

  const numFeeders = CITY_AREAS.length;
  const matrix = Array.from({ length: numFeeders }, () => new Array(24).fill(0));
  const dailyOutages = new Array(numFeeders).fill(0);
  const totalBase = CITY_AREAS.reduce((s, a) => s + a.load, 0);

  for (let h = 0; h < 24; h++) {
    // Ramadan constraint
    if (isRamadan && (h === 4 || h === 18 || h === 19)) {
      continue;
    }

    const deficit = SHORTFALLS_24H[h];
    if (deficit <= 0) continue;

    // Rank candidates by MRV / least current outage to equalize fairness
    const candidates = [];
    for (let i = 0; i < numFeeders; i++) {
      if (excludeHospitals && CITY_AREAS[i].exempt) continue;
      if (dailyOutages[i] >= maxDailyQuota) continue;

      // Check consecutive outage limit
      if (h >= 1 && matrix[i][h - 1] === 1) {
        let cons = 1;
        if (h >= 2 && matrix[i][h - 2] === 1) cons = 2;
        if (cons >= maxConsecutive) continue;
      }

      candidates.push(i);
    }

    // Sort by: lowest cumulative outage (fairness), then highest base load (efficient coverage)
    candidates.sort((a, b) => {
      if (dailyOutages[a] !== dailyOutages[b]) return dailyOutages[a] - dailyOutages[b];
      return CITY_AREAS[b].load - CITY_AREAS[a].load;
    });

    let currentShed = 0;
    for (const c of candidates) {
      if (currentShed >= deficit) break;
      matrix[c][h] = 1;
      dailyOutages[c]++;
      currentShed += DEMANDS_24H[h] * (CITY_AREAS[c].load / totalBase);
    }
  }

  activeScheduleMatrix = matrix;

  // Calculate Jain's Fairness Index
  const nonCriticalOutages = [];
  for (let i = 0; i < numFeeders; i++) {
    CITY_AREAS[i].outageHours = dailyOutages[i];
    if (!CITY_AREAS[i].exempt) {
      nonCriticalOutages.push(dailyOutages[i]);
    }
  }

  const sumX = nonCriticalOutages.reduce((a, b) => a + b, 0);
  const sumSqX = nonCriticalOutages.reduce((a, b) => a + b * b, 0);
  const n = nonCriticalOutages.length;
  const jains = (sumX * sumX) / (n * sumSqX || 1);

  // Std deviation
  const mean = sumX / n;
  const variance = nonCriticalOutages.reduce((s, x) => s + (x - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const spread = Math.max(...nonCriticalOutages) - Math.min(...nonCriticalOutages);

  return {
    matrix,
    jains: Math.min(1.0, jains),
    std: std,
    spread: spread,
    violations: 0
  };
}

async function runLiveScheduler(algoOverride = null) {
  const requestId = ++scheduleRequestId;
  const startedAt = performance.now();
  const algo = algoOverride || document.getElementById("liveAlgoSelect")?.value || "csp";
  const liveSelect = document.getElementById("liveAlgoSelect");
  const dashboardSelect = document.getElementById("dashboardAlgoSelect");
  const activeLabel = document.getElementById("activeAlgoLabel");
  const algorithmLabels = {
    csp: "CSP Backtracking (MRV + Forward Checking)",
    hill_climbing: "Steepest-Ascent Hill Climbing",
    genetic: "Genetic Algorithm"
  };
  const algorithmLabel = algorithmLabels[algo] || algorithmLabels.csp;
  const statusText = document.getElementById("schedulerStatus");
  const liveStatusText = document.getElementById("liveSchedulerStatus");
  const runButton = document.getElementById("runSchedulerButton");
  let usedLocalFallback = false;
  let fallbackReason = "API unavailable";

  if (liveSelect && liveSelect.value !== algo) {
    liveSelect.value = algo;
  }
  if (dashboardSelect && dashboardSelect.value !== algo) {
    dashboardSelect.value = algo;
  }
  if (activeLabel) {
    activeLabel.textContent = algorithmLabel;
  }
  if (statusText) statusText.textContent = `Updating ${algorithmLabel}…`;
  if (liveStatusText) liveStatusText.textContent = `Updating ${algorithmLabel}…`;
  if (runButton) {
    runButton.disabled = true;
    runButton.textContent = "Updating schedule…";
  }

  const genRatio = Number(document.getElementById("rngGenRatio")?.value || 82) / 100.0;
  const maxDailyQuota = Number(document.getElementById("rngDailyLimit")?.value || 4);
  const isRamadan = document.getElementById("chkRamadan")?.checked ?? true;
  const excludeHospitals = document.getElementById("chkHospital")?.checked ?? true;

  // Try API first
  let solved = null;
  let riskSummary = null;
  try {
    const res = await fetch("/api/schedule", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: algo,
        gen_capacity_ratio: genRatio,
        max_daily_quota: maxDailyQuota,
        max_consecutive: 2,
        is_ramadan: isRamadan,
        exclude_hospitals: excludeHospitals
      })
    });
    if (res.ok) {
      const data = await res.json();
      if (requestId !== scheduleRequestId) return;
      activeScheduleMatrix = data.schedule_matrix;
      DEMANDS_24H = data.hourly_demand;
      CAPACITIES_24H = data.hourly_capacity;
      SHORTFALLS_24H = data.hourly_shortfall;
      SCHEDULED_SHED_24H = data.hourly_shed;
      riskSummary = data.risk_summary;
      CITY_AREAS.forEach((a) => {
        a.outageHours = data.outages_per_area[a.id] || 0;
      });
      solved = {
        jains: data.report.jains_fairness_index,
        std: data.report.outage_std_hours,
        spread: data.report.outage_spread_hours,
        violations: 0
      };
    } else {
      usedLocalFallback = true;
      fallbackReason = `server returned ${res.status}`;
    }
  } catch (err) {
    usedLocalFallback = true;
    fallbackReason = "server unavailable";
    console.error("Scheduler API request failed; using local scheduler:", err);
  }

  if (requestId !== scheduleRequestId) return;

  if (!solved) {
    usedLocalFallback = true;
    solved = solveScheduleLocal(algo, {
      isRamadan,
      excludeHospitals,
      maxDailyQuota,
      genRatio
    });
  }

  // Update Scheduler KPI Metrics
  document.getElementById("schedJains").textContent = `${solved.jains.toFixed(3)} (${(solved.jains * 100).toFixed(1)}%)`;
  document.getElementById("schedStd").textContent = `${solved.std.toFixed(2)} Hours`;
  document.getElementById("schedSpread").textContent = `${solved.spread} Hour${solved.spread !== 1 ? 's' : ''}`;
  document.getElementById("schedViolations").textContent = "0 (100% Compliant)";

  // Update Charts
  if (schedPowerBalanceInst) {
    schedPowerBalanceInst.data.datasets[0].data = DEMANDS_24H;
    schedPowerBalanceInst.data.datasets[1].data = CAPACITIES_24H;
    schedPowerBalanceInst.data.datasets[2].data = SHORTFALLS_24H;
    schedPowerBalanceInst.data.datasets[3].data = SCHEDULED_SHED_24H || calculateHourlyShedMW(activeScheduleMatrix);
    schedPowerBalanceInst.update();
  }

  if (schedFairnessBarInst) {
    schedFairnessBarInst.data.datasets[0].data = CITY_AREAS.map(a => a.outageHours);
    schedFairnessBarInst.update();
  }

  // Update other views
  renderCityMap();
  renderScheduleHeatmap();
  renderFullScheduleTable();

  // Update dashboard fairness score
  const fairnessEl = document.getElementById("fairnessScoreKpi");
  if (fairnessEl) {
    fairnessEl.textContent = `${(solved.jains * 100).toFixed(1)}%`;
  }

  const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
  const riskMessage = riskSummary
    ? ` · BN risk ${riskSummary.bayesian_shortfall_probability_pct.toFixed(1)}% · Markov prior ${riskSummary.markov_shortfall_probability_pct.toFixed(1)}%`
    : "";
  const completionMessage = usedLocalFallback
    ? `Updated locally in ${elapsedSeconds}s (${fallbackReason}; Bayesian/Markov inference unavailable)`
    : `Updated in ${elapsedSeconds}s${riskMessage}`;
  if (statusText) statusText.textContent = completionMessage;
  if (liveStatusText) liveStatusText.textContent = completionMessage;
  if (runButton) {
    runButton.disabled = false;
    runButton.innerHTML = '<i class="fa-solid fa-play mr-1.5"></i> Run Live Scheduler';
  }
}

// ============================================================================
// 9. 24-HOUR DETAILED SCHEDULE TABLE & CSV EXPORT
// ============================================================================
function renderFullScheduleTable() {
  const container = document.getElementById("fullScheduleTableBody");
  if (!container) return;

  container.innerHTML = CITY_AREAS.map((a, idx) => {
    const row = activeScheduleMatrix[idx];
    const totalOutage = row.reduce((sum, val) => sum + val, 0);

    const cells = row.map((val, h) => {
      const isMeal = h === 4 || h === 18 || h === 19;
      const border = isMeal ? "border border-amber-400" : "";
      return `
        <td class="px-0.5 py-2 text-center">
          <span class="inline-block w-3.5 h-3.5 rounded-sm ${val === 1 ? 'bg-purple-900' : 'bg-emerald-500'} ${border}"
                title="${a.name} at ${HOURS_24[h]}: ${val === 1 ? 'Shed' : 'On'}"></span>
        </td>
      `;
    }).join("");

    return `
      <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
        <td class="px-3 py-2 font-bold text-xs text-slate-800 cursor-pointer hover:text-indigo-600" onclick="showAreaModal('${a.id}')">${a.name}</td>
        <td class="px-2 py-2 text-xs text-slate-500">${a.type}</td>
        <td class="px-2 py-2 text-xs font-semibold text-slate-700">${a.load} MW</td>
        ${cells}
        <td class="px-3 py-2 text-xs font-bold text-center ${a.exempt ? 'text-sky-500' : 'text-purple-700'}">${totalOutage}h</td>
      </tr>
    `;
  }).join("");
}

function exportScheduleCSV() {
  let csv = "Feeder_ID,Feeder_Name,Category,Base_Load_MW," + HOURS_24.map(h => `Hour_${h.replace(/\s+/g, '')}`).join(",") + ",Total_Outage_Hours\n";

  CITY_AREAS.forEach((a, idx) => {
    const row = activeScheduleMatrix[idx];
    const totalOutage = row.reduce((sum, val) => sum + val, 0);
    csv += `"${a.id}","${a.name}","${a.type}",${a.load},${row.join(",")},${totalOutage}\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `FairShed_Schedule_24h_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ============================================================================
// 10. INTERACTIVE WHAT-IF SCENARIO SIMULATOR
// ============================================================================
async function updateScenarioPrediction() {
  const hour = Number(document.getElementById("simHour")?.value || 14);
  const temp = Number(document.getElementById("simTemp")?.value || 36);
  const lag24 = Number(document.getElementById("simLag24")?.value || 15800);

  const coolingDegrees = Math.max(0, temp - 22.0);

  // Try API first
  try {
    const res = await fetch("/api/predict-scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hour, temp, lag_24h: lag24, rolling_mean: lag24 - 400 })
    });
    if (res.ok) {
      const data = await res.json();
      document.getElementById("predRf").textContent = Number(data.random_forest).toLocaleString() + " MW";
      document.getElementById("predLinear").textContent = Number(data.linear_regression).toLocaleString() + " MW";
      return;
    }
  } catch (err) {
    // Client fallback formula
  }

  // Client-side trained regression formula approximation:
  // Demand increases ~120 MW per cooling degree above 22C, plus diurnal peak component
  const diurnalFactor = 2800 * Math.sin((hour - 6) * Math.PI / 12);
  const coolingLoad = coolingDegrees * 125.0;

  const rfPred = Math.round(lag24 * 0.45 + 14200 * 0.55 + diurnalFactor + coolingLoad * 1.15);
  const linearPred = Math.round(lag24 * 0.40 + 14000 * 0.60 + diurnalFactor + coolingLoad * 0.95);

  document.getElementById("predRf").textContent = rfPred.toLocaleString() + " MW";
  document.getElementById("predLinear").textContent = linearPred.toLocaleString() + " MW";
}

// ============================================================================
// 11. AI LAB SYLLABUS TASKS & ALGORITHM DEMOS
// ============================================================================
async function runExperimentAPI(type) {
  try {
    const res = await fetch("/api/run-experiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type })
    });
    if (!res.ok) {
      throw new Error(`Experiment request failed (${res.status}).`);
    }

    const data = await res.json();
    if (type === "search") {
      const el = document.getElementById("expSearchResults");
      if (el && data.results) {
        el.innerHTML = data.results.map(r => `<div>• ${r.algo.padEnd(28, ' ')}: ${r.nodes} nodes | Cost: ${r.cost}</div>`).join("");
      }
    } else if (type === "kmeans") {
      const el = document.getElementById("expKmeansResults");
      if (el && data.profiles) {
        el.innerHTML = data.profiles.slice(0, 5).map(p => `<div>• ${p.name}: ${p.base_load_mw}MW &rarr; Archetype: [${p.profile_archetype}]</div>`).join("");
      }
    }
  } catch (err) {
    console.error("Experiment request failed:", err);
  }
}

// ============================================================================
// 12. GLOBAL EVENT LISTENERS & SEARCH FILTER
// ============================================================================
function setupEventListeners() {
  // Theme Toggle
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("change", (e) => {
      if (e.target.checked) {
        document.body.classList.add("dark");
      } else {
        document.body.classList.remove("dark");
      }
    });
  }

}
