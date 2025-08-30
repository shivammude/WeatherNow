import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Sun,
  CloudRain,
  Wind,
  Thermometer,
  Loader2,
  MapPin,
  Compass,
  Settings,
  CloudSun,
  MoonStar,
  Umbrella,
} from "lucide-react";

// NOTE: We DO NOT statically import @mlc-ai/web-llm here. Static import can cause
// the bundler to include heavy ML artifacts and can also fail in environments
// that don't support required web backends. Instead we initialize the LLM
// on-demand via dynamic import. If initialization fails we fall back gracefully
// to a non-AI mode so the weather UI remains fully functional.

// ---------- Types ----------
interface GeoPlace {
  id: number;
  name: string;
  country: string;
  admin1?: string;
  latitude: number;
  longitude: number;
}

interface CurrentWeather {
  time: string;
  temperature_2m: number;
  apparent_temperature: number;
  relative_humidity_2m: number;
  precipitation: number;
  rain: number;
  showers: number;
  snowfall: number;
  cloud_cover: number;
  wind_speed_10m: number;
  wind_gusts_10m: number;
  wind_direction_10m: number;
  weather_code: number;
}

// ---------- Helpers ----------
const WMO: Record<number, { label: string; icon: React.ReactNode }> = {
  0: { label: "Clear sky", icon: <Sun className="w-5 h-5" /> },
  1: { label: "Mainly clear", icon: <CloudSun className="w-5 h-5" /> },
  2: { label: "Partly cloudy", icon: <CloudSun className="w-5 h-5" /> },
  3: { label: "Overcast", icon: <CloudSun className="w-5 h-5" /> },
  45: { label: "Fog", icon: <CloudSun className="w-5 h-5" /> },
  48: { label: "Depositing rime fog", icon: <CloudSun className="w-5 h-5" /> },
  51: { label: "Drizzle: light", icon: <CloudRain className="w-5 h-5" /> },
  53: { label: "Drizzle: moderate", icon: <CloudRain className="w-5 h-5" /> },
  55: { label: "Drizzle: dense", icon: <CloudRain className="w-5 h-5" /> },
  61: { label: "Rain: slight", icon: <CloudRain className="w-5 h-5" /> },
  63: { label: "Rain: moderate", icon: <CloudRain className="w-5 h-5" /> },
  65: { label: "Rain: heavy", icon: <CloudRain className="w-5 h-5" /> },
  71: { label: "Snow: slight", icon: <CloudRain className="w-5 h-5" /> },
  73: { label: "Snow: moderate", icon: <CloudRain className="w-5 h-5" /> },
  75: { label: "Snow: heavy", icon: <CloudRain className="w-5 h-5" /> },
  77: { label: "Snow grains", icon: <CloudRain className="w-5 h-5" /> },
  80: { label: "Rain showers: slight", icon: <CloudRain className="w-5 h-5" /> },
  81: { label: "Rain showers: moderate", icon: <CloudRain className="w-5 h-5" /> },
  82: { label: "Rain showers: violent", icon: <CloudRain className="w-5 h-5" /> },
  85: { label: "Snow showers: slight", icon: <CloudRain className="w-5 h-5" /> },
  86: { label: "Snow showers: heavy", icon: <CloudRain className="w-5 h-5" /> },
  95: { label: "Thunderstorm", icon: <CloudRain className="w-5 h-5" /> },
  96: { label: "Thunderstorm w/ hail: slight", icon: <CloudRain className="w-5 h-5" /> },
  99: { label: "Thunderstorm w/ hail: heavy", icon: <CloudRain className="w-5 h-5" /> },
};

const formatWindDir = (deg: number) => {
  if (typeof deg !== "number" || Number.isNaN(deg)) return "—";
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.round(((deg % 360) + 360) / 45) % 8];
};

// Utility to try multiple possible hourly keys (Open-Meteo sometimes uses slightly different names across endpoints)
function pickHourlyValue(hourly: any, keys: string[], index: number) {
  for (const k of keys) {
    if (hourly && Array.isArray(hourly[k]) && typeof hourly[k][index] !== "undefined") return hourly[k][index];
  }
  return undefined;
}

// ---------- Main App ----------
export default function WeatherNowApp() {
  const [query, setQuery] = useState("");
  const [places, setPlaces] = useState<GeoPlace[]>([]);
  const [loadingSearch, setLoadingSearch] = useState(false);

  const [selected, setSelected] = useState<GeoPlace | null>(null);
  const [current, setCurrent] = useState<CurrentWeather | null>(null);
  const [units, setUnits] = useState({ temp: "celsius", wind: "kmh" });
  const [fetching, setFetching] = useState(false);

  // LLM state: we initialize only when user asks for AI (or when explicitly triggered).
  // This avoids automatic failures in environments that don't support web backends.
  const [llmStatus, setLlmStatus] = useState<"idle" | "loading" | "ready" | "failed">("idle");
  const [llmError, setLlmError] = useState<string | null>(null);
  const [llmThinking, setLlmThinking] = useState(false);
  const [advice, setAdvice] = useState<string>("");
  const engineRef = useRef<any>(null);

  // Debounce & request-tracking for live autocomplete
  const searchTimer = useRef<number | null>(null);
  const searchReqId = useRef(0);

  // Start LLM on demand (user clicks). Uses dynamic import to avoid bundling & to gracefully handle environments
  // where the library or required web features aren't available.
  const initLLM = async () => {
    if (llmStatus === "loading" || llmStatus === "ready") return;
    setLlmStatus("loading");
    setLlmError(null);

    try {
      // Basic feature check: WebGPU is ideal but not always required; dynamic import catches missing module too
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = await import("@mlc-ai/web-llm");
      if (!mod || !mod.CreateMLCEngine) throw new Error("WebLLM library not available in this environment");

      const model = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"; // small-ish instruct model
      const engine = await mod.CreateMLCEngine(model, {
        initProgressCallback: (p: any) => {
          if (p?.text) console.log("LLM init:", p.text);
        },
      });

      engineRef.current = engine;
      setLlmStatus("ready");
    } catch (e: any) {
      console.warn("LLM init failed:", e);
      setLlmError(e?.message ? String(e.message) : String(e));
      setLlmStatus("failed");
    }
  };

  // Live-search when user types (debounced)
  useEffect(() => {
    if (searchTimer.current) {
      window.clearTimeout(searchTimer.current);
      searchTimer.current = null;
    }
    if (!query || query.trim().length < 2) {
      setPlaces([]);
      return;
    }

    const q = query.trim();
    searchTimer.current = window.setTimeout(() => {
      (async () => {
        setLoadingSearch(true);
        const reqId = ++searchReqId.current;
        try {
          const url = new URL("https://geocoding-api.open-meteo.com/v1/search");
          url.searchParams.set("name", q);
          url.searchParams.set("count", "8");
          url.searchParams.set("language", "en");

          const res = await fetch(url.toString());
          const json = await res.json();

          // Only accept this response if it's the latest request
          if (reqId !== searchReqId.current) return;

          const list: GeoPlace[] = (json.results || []).map((r: any) => ({
            id: r.id ?? Math.floor(Math.random() * 1e9),
            name: r.name,
            country: r.country,
            admin1: r.admin1,
            latitude: r.latitude,
            longitude: r.longitude,
          }));
          setPlaces(list);
        } catch (e) {
          console.warn("Geocoding lookup failed", e);
        } finally {
          setLoadingSearch(false);
        }
      })();
    }, 350);

    return () => {
      if (searchTimer.current) window.clearTimeout(searchTimer.current);
    };
  }, [query]);

  // Fetch weather when a place is selected or units change
  useEffect(() => {
    const fetchWeather = async () => {
      if (!selected) return;
      setFetching(true);
      setAdvice("");
      try {
        const url = new URL("https://api.open-meteo.com/v1/forecast");
        url.searchParams.set("latitude", String(selected.latitude));
        url.searchParams.set("longitude", String(selected.longitude));
        // Ask for current_weather plus hourly variables so we can pick matching current values
        url.searchParams.set("current_weather", "true");
        url.searchParams.set(
          "hourly",
          "relativehumidity_2m,apparent_temperature,cloudcover,precipitation,windgusts_10m,showers,snowfall"
        );
        url.searchParams.set("timezone", "auto");
        url.searchParams.set("temperature_unit", units.temp);
        // Note: Open-Meteo expects windspeed_unit (not wind_speed_unit)
        url.searchParams.set("windspeed_unit", units.wind);

        const res = await fetch(url.toString());
        const data = await res.json();

        const cw = data.current_weather || {};
        const hourly = data.hourly || {};

        // Find index of current hour in hourly.time if possible
        let idx = -1;
        if (Array.isArray(hourly.time) && cw.time) idx = hourly.time.indexOf(cw.time);
        if (idx === -1 && Array.isArray(hourly.time) && hourly.time.length > 0) idx = hourly.time.length - 1; // fallback to latest

        const c: CurrentWeather = {
          time: cw.time || "",
          temperature_2m: (cw.temperature ?? (Array.isArray(hourly.temperature_2m) ? hourly.temperature_2m[idx] : undefined) ?? 0) as number,
          apparent_temperature:
            (cw.apparent_temperature ?? pickHourlyValue(hourly, ["apparent_temperature", "apparentTemperature"], idx) ?? cw.temperature ?? 0) as number,
          relative_humidity_2m: (pickHourlyValue(hourly, ["relativehumidity_2m", "relative_humidity_2m"], idx) ?? 0) as number,
          precipitation: (pickHourlyValue(hourly, ["precipitation"], idx) ?? 0) as number,
          rain: (pickHourlyValue(hourly, ["rain"], idx) ?? 0) as number,
          showers: (pickHourlyValue(hourly, ["showers"], idx) ?? 0) as number,
          snowfall: (pickHourlyValue(hourly, ["snowfall"], idx) ?? 0) as number,
          cloud_cover: (pickHourlyValue(hourly, ["cloudcover", "cloud_cover"], idx) ?? cw.cloudcover ?? 0) as number,
          wind_speed_10m: (cw.windspeed ?? pickHourlyValue(hourly, ["windspeed_10m", "wind_speed_10m"], idx) ?? 0) as number,
          wind_gusts_10m: (cw.windgusts ?? pickHourlyValue(hourly, ["windgusts_10m", "wind_gusts_10m"], idx) ?? 0) as number,
          wind_direction_10m: (cw.winddirection ?? pickHourlyValue(hourly, ["winddirection_10m", "wind_direction_10m"], idx) ?? 0) as number,
          weather_code: (cw.weathercode ?? cw.weather_code ?? 0) as number,
        };

        setCurrent(c);

        // Trigger AI advice only if engine ready
        if (engineRef.current) generateAdvice(c, selected);
      } catch (e) {
        console.error("Weather fetch failed", e);
      } finally {
        setFetching(false);
      }
    };

    fetchWeather();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, units]);

  const unitLabel = useMemo(
    () => ({
      temp: units.temp === "celsius" ? "°C" : "°F",
      wind: units.wind === "kmh" ? "km/h" : units.wind === "mph" ? "mph" : units.wind === "ms" ? "m/s" : "kn",
    }),
    [units]
  );

  async function generateAdvice(c: CurrentWeather, place: GeoPlace) {
    if (!engineRef.current) return;
    setLlmThinking(true);
    try {
      const prompt = `You are an outdoors guide. Given current weather, craft a concise, friendly brief for Jamie (an outdoor enthusiast). Include: (1) one-line summary, (2) suggested activities (3-5), (3) safety + packing tips (bullets), and (4) a go/no-go indicator (with emoji).\n\nCity: ${place.name}${place.admin1 ? ", " + place.admin1 : ""}${place.country ? ", " + place.country : ""}\nTemp: ${c.temperature_2m}${unitLabel.temp} (feels like ${c.apparent_temperature}${unitLabel.temp})\nHumidity: ${c.relative_humidity_2m}%\nWind: ${c.wind_speed_10m} ${unitLabel.wind} (gusts ${c.wind_gusts_10m} ${unitLabel.wind}) from ${formatWindDir(c.wind_direction_10m)}\nClouds: ${c.cloud_cover}%\nPrecip: ${c.precipitation} mm (rain ${c.rain} mm, showers ${c.showers} mm, snow ${c.snowfall} cm)\nCondition: ${(WMO[c.weather_code]?.label) || "Unknown"}.\n\nKeep it under 120 words, crisp bullets for tips.`;

      // The engine API surface can vary by version; be defensive when calling it.
      // @ts-ignore dynamic engine
      const reply = await engineRef.current?.chat?.completions?.create?.({
        messages: [
          { role: "system", content: "Be concise, helpful, and upbeat." },
          { role: "user", content: prompt },
        ],
        temperature: 0.6,
        max_tokens: 220,
      });

      const text = reply?.choices?.[0]?.message?.content?.trim?.() || "";
      setAdvice(text);
    } catch (e) {
      console.warn("LLM generate failed", e);
      setAdvice("");
    } finally {
      setLlmThinking(false);
    }
  }

  // Geolocation quick-pick
  const useMyLocation = async () => {
    if (!navigator.geolocation) return alert("Geolocation not supported");
    navigator.geolocation.getCurrentPosition(async (pos) => {
      const { latitude, longitude } = pos.coords;
      setSelected({
        id: Date.now(),
        name: "My Location",
        country: "",
        latitude,
        longitude,
      });
      setQuery("My Location");
      setPlaces([]);
    });
  };

  // Simple self-tests (run in dev) - adds a couple of tiny checks to help future debugging
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") {
      console.assert(formatWindDir(0) === "N", "formatWindDir(0) should be N");
      console.assert(typeof pickHourlyValue === "function", "pickHourlyValue must exist");
    }
  }, []);

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800 text-slate-100 p-4 md:p-8">
      <header className="max-w-3xl mx-auto flex items-center gap-3">
        <div className="p-2 rounded-2xl bg-slate-800 shadow-lg shadow-slate-900/40">
          <Sun className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Weather Now</h1>
          <p className="text-slate-300 text-sm">Open‑Meteo + optional in‑browser LLM suggestions for Jamie</p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setUnits((u) => ({ ...u, temp: u.temp === "celsius" ? "fahrenheit" : "celsius" }))}
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 transition shadow"
            title="Toggle °C/°F"
          >
            {units.temp === "celsius" ? "°C" : "°F"}
          </button>

          <button
            onClick={() =>
              setUnits((u) => ({
                ...u,
                wind: u.wind === "kmh" ? "mph" : u.wind === "mph" ? "ms" : u.wind === "ms" ? "kn" : "kmh",
              }))
            }
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 transition shadow flex items-center gap-2"
            title="Cycle wind unit"
          >
            <Wind className="w-4 h-4" /> {unitLabel.wind}
          </button>

          <div className="px-2 py-1.5 rounded-xl bg-slate-800 text-xs uppercase tracking-wide flex items-center gap-2" title="LLM status">
            <Settings className={`w-4 h-4 ${llmThinking ? "animate-spin" : ""}`} />
            {llmStatus === "idle" && <button onClick={initLLM} className="underline">Enable AI</button>}
            {llmStatus === "loading" && <span>loading AI…</span>}
            {llmStatus === "ready" && <span>AI ready</span>}
            {llmStatus === "failed" && <span title={llmError ?? "failed"}>AI unavailable</span>}
          </div>
        </div>
      </header>

      {/* Search */}
      <section className="max-w-3xl mx-auto mt-6">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search any city (type 2+ chars — e.g., Pune, Denver, Tokyo)"
              className="w-full bg-slate-800/70 backdrop-blur px-4 py-3 rounded-2xl outline-none ring-1 ring-slate-700 focus:ring-indigo-500 transition placeholder:text-slate-400"
              onKeyDown={(e) => {
                if (e.key === "Enter" && query.trim() && places.length > 0) {
                  // pick first suggestion on Enter
                  const first = places[0];
                  setSelected(first);
                  setPlaces([]);
                }
              }}
            />
            <Search className="w-5 h-5 absolute right-3 top-1/2 -translate-y-1/2 text-slate-400" />
          </div>

          <button
            onClick={useMyLocation}
            className="px-4 py-3 rounded-2xl bg-slate-800 hover:bg-slate-700 shadow flex items-center gap-2"
            title="Use my location"
          >
            <Compass className="w-4 h-4" />
            Near me
          </button>
        </div>

        <AnimatePresence>
          {places.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              className="mt-3 grid gap-2"
            >
              {places.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setSelected(p);
                    setPlaces([]);
                    setQuery(p.name + (p.admin1 ? ", " + p.admin1 : ""));
                  }}
                  className="text-left w-full px-4 py-3 rounded-2xl bg-slate-800 hover:bg-slate-700 transition flex items-center gap-3"
                >
                  <MapPin className="w-4 h-4" />
                  <div className="flex-1">
                    <div className="font-medium">{p.name}</div>
                    <div className="text-sm text-slate-400">{[p.admin1, p.country].filter(Boolean).join(", ")}</div>
                  </div>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* Result */}
      <section className="max-w-3xl mx-auto mt-6">
        <div className="grid gap-4">
          <AnimatePresence>
            {fetching && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="p-6 rounded-2xl bg-slate-800/60 ring-1 ring-slate-700 flex items-center gap-3"
              >
                <Loader2 className="w-5 h-5 animate-spin" /> Fetching latest weather…
              </motion.div>
            )}
          </AnimatePresence>

          {selected && current && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl bg-slate-800 ring-1 ring-slate-700 p-6 shadow-xl"
            >
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-2xl bg-slate-900/60">{WMO[current.weather_code]?.icon || <Sun className="w-6 h-6" />}</div>
                <div className="flex-1">
                  <div className="text-sm text-slate-400">{selected.name}{selected.admin1 ? `, ${selected.admin1}` : ""}{selected.country ? `, ${selected.country}` : ""}</div>
                  <div className="text-3xl font-semibold mt-1 flex items-center gap-3">
                    <Thermometer className="w-6 h-6" /> {current.temperature_2m}
                    <span className="text-slate-300 text-xl">{unitLabel.temp}</span>
                    <span className="text-slate-400 text-base">(feels {current.apparent_temperature}{unitLabel.temp})</span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <Stat label="Condition" value={WMO[current.weather_code]?.label || "—"} icon={<Sun className="w-4 h-4" />} />
                    <Stat label="Humidity" value={`${current.relative_humidity_2m}%`} icon={<CloudSun className="w-4 h-4" />} />
                    <Stat label="Wind" value={`${current.wind_speed_10m} ${unitLabel.wind}`} icon={<Wind className="w-4 h-4" />} />
                    <Stat label="Gusts" value={`${current.wind_gusts_10m} ${unitLabel.wind}`} icon={<Wind className="w-4 h-4" />} />
                    <Stat label="Direction" value={`${formatWindDir(current.wind_direction_10m)} (${current.wind_direction_10m}°)`} icon={<Compass className="w-4 h-4" />} />
                    <Stat label="Clouds" value={`${current.cloud_cover}%`} icon={<CloudSun className="w-4 h-4" />} />
                    <Stat label="Precip (1h)" value={`${current.precipitation} mm`} icon={<Umbrella className="w-4 h-4" />} />
                  </div>
                </div>
              </div>

              {/* LLM Advisory */}
              <div className="mt-6 p-4 rounded-2xl bg-slate-900/60 ring-1 ring-slate-700/60">
                <div className="flex items-center gap-2 text-slate-300">
                  <MoonStar className="w-4 h-4" />
                  <span className="font-medium">Jamie’s Outdoor Brief (AI)</span>
                </div>
                <div className="mt-3 whitespace-pre-wrap text-slate-200 text-sm leading-6">
                  {llmStatus === "idle" && <span className="opacity-70">Enable AI to get activity suggestions and safety tips.</span>}
                  {llmStatus === "loading" && <span className="opacity-80">Setting up AI…</span>}
                  {llmStatus === "failed" && (
                    <span className="opacity-80">AI unavailable: {llmError ?? "initialization failed"}. You can still use the weather features.</span>
                  )}
                  {llmThinking && <span className="opacity-80">Crafting advice…</span>}
                  {!llmThinking && !llmError && advice && advice}
                  {!llmThinking && llmStatus === "ready" && !advice && (
                    <span className="opacity-70">AI is ready — select a city to generate advice.</span>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="max-w-3xl mx-auto mt-10 text-xs text-slate-400 flex flex-wrap gap-2 items-center">
        <span>Data: Open‑Meteo (no key required)</span>
        <span>•</span>
        <span>AI: optional WebLLM in‑browser (no server, no key)</span>
        <span>•</span>
        <a className="underline decoration-dotted" href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open‑Meteo Docs</a>
        <span>•</span>
        <a className="underline decoration-dotted" href="https://github.com/mlc-ai/web-llm" target="_blank" rel="noreferrer">WebLLM</a>
      </footer>
    </div>
  );
}

function Stat({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return (
    <div className="p-3 rounded-xl bg-slate-900/60 ring-1 ring-slate-700/60 flex items-center gap-2">
      <span className="text-slate-400" title={label}>{icon}</span>
      <div>
        <div className="text-[11px] uppercase tracking-wide text-slate-400">{label}</div>
        <div className="text-sm font-medium text-slate-100">{value}</div>
      </div>
    </div>
  );
}

// End of file - key fixes made:
// 1) LLM init moved to on-demand dynamic import (no automatic init on mount).
// 2) Autocomplete is live (debounced) as the user types (2+ chars).
// 3) Weather fetch carefully maps fields from Open-Meteo's current_weather + hourly arrays,
//    handling different possible key names and avoiding runtime crashes.
// 4) Friendly AI status + graceful fallback so the rest of the app keeps working even if LLM fails.
