const PUBLIC_LOCATIONS = {
  F01_HOSP: [24.8739, 67.0281], F02_IND: [24.8827, 66.9983],
  F03_COMM_A: [24.8587, 67.0104], F04_RES_A: [24.8068, 67.0577],
  F05_RES_B: [24.9165, 67.0959], F06_SUB_A: [24.9472, 67.0335],
  F07_SUB_B: [24.8896, 67.1695], F08_RUR: [25.0610, 67.2480],
  F09_COMM_B: [24.8662, 67.0587], F10_RES_C: [24.8138, 67.0300]
};

let publicData = null;
let publicMap = null;
let publicMarkers = new Map();
let activeFilter = "all";
let pendingLocality = "all";
let selectedLocality = "all";
let countdownTimer = null;

function getStatus(feeder) {
  if (feeder.exempt) return { label: "Protected", className: "protected" };
  if (feeder.outage_hours) return { label: `${feeder.outage_hours}h planned`, className: "outage" };
  return { label: "Power available", className: "online" };
}

function selectLocality(id) {
  const feeder = publicData.feeders.find((item) => item.id === id);
  const name = document.getElementById("publicSelectedName");
  const summary = document.getElementById("publicSelectedSummary");
  const badge = document.getElementById("publicSelectedBadge");
  const details = document.getElementById("publicSelectedDetails");
  const tools = document.getElementById("publicLocalityTools");
  if (!feeder) {
    selectedLocality = "all";
    name.textContent = "All Karachi localities";
    summary.textContent = "Select a locality above to see its public status.";
    badge.textContent = "Overview";
    details.classList.add("hidden");
    tools.classList.add("hidden");
    if (countdownTimer) clearInterval(countdownTimer);
    publicMap.fitBounds(L.latLngBounds(Object.values(PUBLIC_LOCATIONS)), { padding: [18, 18], maxZoom: 11, animate: true });
    return;
  }
  selectedLocality = id;
  tools.classList.remove("hidden");
  const status = getStatus(feeder);
  name.textContent = feeder.name;
  summary.textContent = `${feeder.category} service area · ${status.label}`;
  badge.textContent = status.label;
  badge.className = `public-status-badge ${status.className}`;
  details.classList.remove("hidden");
  details.innerHTML = `
    <div><p class="public-detail-label">Service status</p><p class="public-detail-value">${status.className === "outage" ? "Planned interruption" : "Power available"}</p></div>
    <div><p class="public-detail-label">Outage time</p><p class="public-detail-value">${feeder.outage_hours} hours</p></div>
    <div><p class="public-detail-label">Area type</p><p class="public-detail-value">${feeder.category}</p></div>
    <div><p class="public-detail-label">Priority</p><p class="public-detail-value">${feeder.exempt ? "Critical protected" : "Standard service"}</p></div>`;
  const marker = publicMarkers.get(id);
  if (marker) {
    publicMap.setView(marker.getLatLng(), 13, { animate: true });
    marker.openPopup();
  }
  updateCountdown();
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(updateCountdown, 1000);
}

function getSelectedSchedule() {
  const index = publicData.feeders.findIndex((feeder) => feeder.id === selectedLocality);
  return index >= 0 ? publicData.schedule_matrix[index] : null;
}

function updateCountdown() {
  if (!publicData || selectedLocality === "all") return;
  const feeder = publicData.feeders.find((item) => item.id === selectedLocality);
  const row = getSelectedSchedule();
  const label = document.getElementById("publicCountdownLabel");
  const value = document.getElementById("publicCountdown");
  const note = document.getElementById("publicCountdownNote");
  if (!feeder || !row || feeder.exempt || !row.includes(1)) {
    label.textContent = "No planned interruption is scheduled.";
    value.textContent = "ONLINE";
    note.textContent = feeder?.exempt ? "This critical service area is protected." : "Power is available for this locality.";
    return;
  }
  const now = new Date();
  const currentHour = now.getHours();
  let targetHour = row[currentHour] ? currentHour + 1 : -1;
  let isActive = row[currentHour] === 1;
  if (targetHour < 0) {
    for (let offset = 1; offset <= 24; offset++) {
      const candidate = (currentHour + offset) % 24;
      if (row[candidate] === 1) {
        targetHour = candidate;
        break;
      }
    }
  }
  if (targetHour < 0) {
    label.textContent = "No upcoming planned interruption.";
    value.textContent = "ONLINE";
    note.textContent = "Power is available for this locality.";
    return;
  }
  const target = new Date(now);
  target.setHours(targetHour, isActive ? 0 : 0, 0, 0);
  if (!isActive && target <= now) target.setDate(target.getDate() + 1);
  let seconds = Math.max(0, Math.floor((target - now) / 1000));
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  seconds %= 3600;
  const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
  const remaining = String(seconds % 60).padStart(2, "0");
  value.textContent = `${hours}:${minutes}:${remaining}`;
  label.textContent = isActive ? "Estimated time until planned outage ends" : "Time until next planned outage";
  note.textContent = isActive ? "This is an hourly planning estimate." : `Next planned hour: ${publicData.hours[targetHour % 24]}`;
}

function answerPublicQuery(question) {
  const feeder = publicData.feeders.find((item) => item.id === selectedLocality);
  const row = getSelectedSchedule();
  const normalized = question.toLowerCase();
  if (!feeder || !row) return "Please select a specific locality first.";
  const outageHours = row.map((value, hour) => value ? publicData.hours[hour] : null).filter(Boolean);
  if (normalized.includes("when") || normalized.includes("schedule") || normalized.includes("time")) {
    return outageHours.length ? `${feeder.name} has planned interruptions at ${outageHours.join(", ")}.` : `${feeder.name} has no planned interruption in today's schedule.`;
  }
  if (normalized.includes("why") || normalized.includes("reason")) {
    return feeder.exempt ? `${feeder.name} is marked as a critical service area and is protected.` : "Planned interruptions help balance available generation with city demand.";
  }
  if (normalized.includes("status") || normalized.includes("power") || normalized.includes("outage")) {
    return feeder.exempt ? `${feeder.name} is protected from planned load shedding.` : `${feeder.name} has ${feeder.outage_hours} planned outage hour(s) today.`;
  }
  return "I can answer about this locality's status, outage schedule, outage time, and protection.";
}

function renderPublicStatus(data) {
  publicData = data;
  const cards = document.getElementById("publicAreaCards");
  const select = document.getElementById("publicLocalitySelect");
  select.innerHTML = '<option value="all">All Karachi localities</option>' +
    data.feeders.map((feeder) => `<option value="${feeder.id}">${feeder.name}</option>`).join("");
  select.value = pendingLocality;
  const outageCount = data.feeders.filter((feeder) => feeder.outage_hours > 0).length;
  document.getElementById("publicOnlineCount").textContent = `${data.feeders.length - outageCount}/${data.feeders.length}`;
  document.getElementById("publicOutageCount").textContent = outageCount;
  document.getElementById("publicProtectedCount").textContent = data.feeders.filter((feeder) => feeder.exempt).length;
  document.getElementById("publicUpdatedAt").textContent = new Date(data.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  renderCards();
  renderSchedule();
}

function renderCards() {
  const cards = document.getElementById("publicAreaCards");
  const visible = publicData.feeders.filter((feeder) => activeFilter === "all" ||
    (activeFilter === "outage" && feeder.outage_hours > 0) ||
    (activeFilter === "protected" && feeder.exempt));
  cards.innerHTML = visible.map((feeder) => {
    const status = getStatus(feeder);
    return `<button type="button" class="public-area-card ${status.className}" data-locality="${feeder.id}">
      <span class="flex items-center justify-between gap-2"><span class="font-bold text-sm">${feeder.name}</span><span class="public-status-dot ${status.className}"></span></span>
      <span class="block text-xs text-slate-500 mt-1">${feeder.category}</span>
      <span class="flex items-end justify-between mt-4"><span class="text-xl font-extrabold">${feeder.outage_hours}h</span><span class="text-[11px] font-bold">${status.label}</span></span>
    </button>`;
  }).join("") || '<p class="text-sm text-slate-500 col-span-full">No localities match this filter.</p>';
  cards.querySelectorAll("[data-locality]").forEach((card) =>
    card.addEventListener("click", () => {
      pendingLocality = card.dataset.locality;
    })
  );
}

function renderSchedule() {
  const timeline = document.getElementById("publicScheduleTimeline");
  const table = document.getElementById("publicScheduleBody");
  timeline.innerHTML = publicData.hours.map((hour, index) =>
    `<span class="public-hour ${publicData.schedule_matrix.some((row) => row[index]) ? "active" : ""}">${hour.replace(":00", "")}</span>`
  ).join("");
  table.innerHTML = publicData.feeders.map((feeder, index) => {
    const row = publicData.schedule_matrix[index];
    const hours = row.map((value, hour) => value ? publicData.hours[hour] : null).filter(Boolean);
    return `<button type="button" class="public-schedule-row" data-locality="${feeder.id}">
      <span><strong>${feeder.name}</strong><small>${feeder.category}</small></span>
      <span class="text-right">${hours.length ? hours.slice(0, 3).join(", ") + (hours.length > 3 ? " ..." : "") : "No planned outage"}<small>${hours.length} planned hour${hours.length === 1 ? "" : "s"}</small></span>
    </button>`;
  }).join("");
    table.querySelectorAll("[data-locality]").forEach((row) =>
      row.addEventListener("click", () => {
        pendingLocality = row.dataset.locality;
      })
    );
}

function renderPublicMap(feeders) {
  publicMap = L.map("publicMap", { attributionControl: false }).setView([24.88, 67.06], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18 }).addTo(publicMap);
  const locations = [];
  feeders.forEach((feeder) => {
    const location = PUBLIC_LOCATIONS[feeder.id];
    if (!location) return;
    locations.push(location);
    const marker = L.marker(location).addTo(publicMap).bindPopup(`<strong>${feeder.name}</strong><br>${getStatus(feeder).label}`);
    marker.on("click", () => {
      pendingLocality = feeder.id;
    });
    publicMarkers.set(feeder.id, marker);
  });
  const reset = () => publicMap.fitBounds(L.latLngBounds(locations), { padding: [18, 18], maxZoom: 11, animate: true });
  document.getElementById("publicMapReset")?.addEventListener("click", reset);
  setTimeout(() => { publicMap.invalidateSize({ pan: false }); reset(); }, 0);
}

document.querySelectorAll(".public-filter").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".public-filter").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  activeFilter = button.dataset.filter;
  renderCards();
}));
document.getElementById("publicLocalitySelect")?.addEventListener("change", (event) => {
  pendingLocality = event.target.value;
});
document.getElementById("publicLocateButton")?.addEventListener("click", () => {
  selectLocality(pendingLocality);
  document.getElementById("publicSelectedName").scrollIntoView({ behavior: "smooth", block: "center" });
});
document.getElementById("publicQueryForm")?.addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("publicQueryInput");
  document.getElementById("publicQueryAnswer").textContent = answerPublicQuery(input.value.trim());
});

fetch("/api/public-status")
  .then((response) => { if (!response.ok) throw new Error("Public status is unavailable."); return response.json(); })
  .then((data) => { renderPublicStatus(data); renderPublicMap(data.feeders); })
  .catch((error) => { document.getElementById("publicUpdatedAt").textContent = error.message; });
