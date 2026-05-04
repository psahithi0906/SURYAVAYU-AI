import { useEffect, useMemo, useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const fallbackPlants = [
  { plant_id: 'SOLAR_001', name: 'Pavagada Solar Park', plant_type: 'solar', lat: 14.099, lng: 77.28, capacity_mw: 2050, cluster_id: 'Tumakuru-Solar' },
  { plant_id: 'SOLAR_002', name: 'Raichur Solar Plains', plant_type: 'solar', lat: 16.212, lng: 77.3439, capacity_mw: 600, cluster_id: 'Raichur-Solar' },
  { plant_id: 'SOLAR_003', name: 'Koppal Solar Farm', plant_type: 'solar', lat: 15.348, lng: 76.154, capacity_mw: 350, cluster_id: 'Koppal-Solar' },
  { plant_id: 'SOLAR_004', name: 'Kalaburagi Solar Zone', plant_type: 'solar', lat: 17.3297, lng: 76.8343, capacity_mw: 520, cluster_id: 'North-Solar' },
  { plant_id: 'SOLAR_005', name: 'Yadgir Solar Field', plant_type: 'solar', lat: 16.77, lng: 77.1376, capacity_mw: 300, cluster_id: 'North-Solar' },
  { plant_id: 'WIND_001', name: 'Chitradurga Wind Farm', plant_type: 'wind', lat: 14.2306, lng: 76.398, capacity_mw: 450, cluster_id: 'Central-Wind' },
  { plant_id: 'WIND_002', name: 'Gadag Wind Corridor', plant_type: 'wind', lat: 15.431, lng: 75.635, capacity_mw: 320, cluster_id: 'Gadag-Wind' },
  { plant_id: 'WIND_003', name: 'Vijayapura Wind Zone', plant_type: 'wind', lat: 16.8302, lng: 75.71, capacity_mw: 280, cluster_id: 'North-Wind' },
  { plant_id: 'WIND_004', name: 'Belagavi Wind Ridge', plant_type: 'wind', lat: 15.8497, lng: 74.4977, capacity_mw: 500, cluster_id: 'Western-Wind' },
  { plant_id: 'HYBRID_001', name: 'Ballari Hybrid Park', plant_type: 'hybrid', lat: 15.1394, lng: 76.9214, capacity_mw: 700, cluster_id: 'Ballari-Hybrid' },
];

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function formatHour(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function ForecastChart({ points, actuals, capacity }) {
  const width = 880;
  const height = 320;
  const pad = { top: 24, right: 28, bottom: 42, left: 58 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const values = points.flatMap((point) => [point.p10, point.p50, point.p90]).concat(actuals.map((point) => point.actual_mw));
  const maxY = Math.max(capacity || 1, ...values, 1) * 1.04;
  const x = (index) => pad.left + (points.length <= 1 ? 0 : (index / (points.length - 1)) * chartW);
  const y = (value) => pad.top + chartH - (value / maxY) * chartH;
  const p50Path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(index)} ${y(point.p50)}`).join(' ');
  const upper = points.map((point, index) => `${x(index)} ${y(point.p90)}`).join(' L ');
  const lower = points.slice().reverse().map((point, reverseIndex) => `${x(points.length - 1 - reverseIndex)} ${y(point.p10)}`).join(' L ');
  const bandPath = points.length ? `M ${upper} L ${lower} Z` : '';

  return (
    <div className="chartShell">
      <div className="panelTitleRow">
        <div>
          <p className="eyebrow">Forecast envelope</p>
          <h2>Next {points.length || 0} hours, MW output</h2>
        </div>
        <div className="legend">
          <span><i className="bandSwatch" /> P10-P90</span>
          <span><i className="lineSwatch" /> P50</span>
          <span><i className="dotSwatch" /> Actual</span>
        </div>
      </div>
      <svg className="forecastSvg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Probabilistic forecast chart">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={pad.left} x2={width - pad.right} y1={pad.top + chartH * tick} y2={pad.top + chartH * tick} className="gridLine" />
            <text x={18} y={pad.top + chartH * tick + 5} className="axisText">{Math.round(maxY * (1 - tick))}</text>
          </g>
        ))}
        <path d={bandPath} className="uncertaintyBand" />
        <path d={p50Path} className="medianLine" />
        {actuals.map((point, index) => (
          <circle key={point.timestamp} cx={x(index)} cy={y(point.actual_mw)} r="4.5" className="actualDot" />
        ))}
        {points.filter((_, index) => index % Math.ceil(points.length / 6 || 1) === 0).map((point, index) => {
          const originalIndex = points.indexOf(point);
          return <text key={`${point.timestamp}-${index}`} x={x(originalIndex)} y={height - 12} className="xText">{formatHour(point.timestamp)}</text>;
        })}
      </svg>
    </div>
  );
}

function ShapChart({ drivers }) {
  const maxAbs = Math.max(...drivers.map((driver) => Math.abs(driver.value)), 0.1);
  return (
    <div className="panel shapPanel">
      <div className="panelTitleRow">
        <div>
          <p className="eyebrow">Explainability</p>
          <h2>Top forecast drivers</h2>
        </div>
      </div>
      <div className="driverList">
        {drivers.map((driver) => {
          const width = `${Math.max(8, (Math.abs(driver.value) / maxAbs) * 48)}%`;
          return (
            <div className="driverRow" key={driver.feature}>
              <span>{driver.feature}</span>
              <div className="driverTrack">
                <b className={driver.value >= 0 ? 'positive' : 'negative'} style={{ width }} />
              </div>
              <strong>{driver.value > 0 ? '+' : ''}{driver.value.toFixed(2)}</strong>
            </div>
          );
        })}
      </div>
      <p className="insight">
        {drivers[0]?.feature || 'Weather'} is the strongest current signal; positive bars lift generation and negative bars suppress it.
      </p>
    </div>
  );
}

function PlantMap({ plants, selectedPlantId, onSelect }) {
  const latMin = 13.6;
  const latMax = 18.2;
  const lngMin = 74.0;
  const lngMax = 77.9;
  const left = (lng) => `${((lng - lngMin) / (lngMax - lngMin)) * 100}%`;
  const top = (lat) => `${(1 - (lat - latMin) / (latMax - latMin)) * 100}%`;

  return (
    <div className="panel mapPanel">
      <div className="panelTitleRow">
        <div>
          <p className="eyebrow">Karnataka grid</p>
          <h2>Renewable assets</h2>
        </div>
      </div>
      <div className="mapCanvas">
        <div className="stateShape" />
        {plants.map((plant) => (
          <button
            type="button"
            key={plant.plant_id}
            className={`plantMarker ${plant.plant_type} ${selectedPlantId === plant.plant_id ? 'active' : ''}`}
            style={{ left: left(plant.lng), top: top(plant.lat) }}
            onClick={() => onSelect(plant.plant_id)}
            title={`${plant.name} - ${plant.capacity_mw} MW`}
          >
            {plant.plant_type === 'wind' ? 'W' : plant.plant_type === 'hybrid' ? 'H' : 'S'}
          </button>
        ))}
      </div>
    </div>
  );
}

function KpiStrip({ forecast }) {
  const total = forecast?.points?.reduce((sum, point) => sum + point.p50, 0) || 0;
  const peak = Math.max(...(forecast?.points || [{ p50: 0 }]).map((point) => point.p50), 0);
  const metrics = forecast?.metrics || {};
  return (
    <section className="kpiGrid">
      <div className="kpi"><span>Expected energy</span><strong>{total.toFixed(0)} MWh</strong></div>
      <div className="kpi"><span>Peak output</span><strong>{peak.toFixed(0)} MW</strong></div>
      <div className="kpi"><span>NRMSE</span><strong>{((metrics.nrmse || 0) * 100).toFixed(1)}%</strong></div>
      <div className="kpi"><span>P10-P90 coverage</span><strong>{((metrics.coverage_p10_p90 || 0) * 100).toFixed(1)}%</strong></div>
    </section>
  );
}

function App() {
  const [plants, setPlants] = useState(fallbackPlants);
  const [selectedPlantId, setSelectedPlantId] = useState('SOLAR_001');
  const [targetDate, setTargetDate] = useState(todayIso());
  const [horizon, setHorizon] = useState(24);
  const [forecast, setForecast] = useState(null);
  const [actuals, setActuals] = useState([]);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');

  const selectedPlant = useMemo(
    () => plants.find((plant) => plant.plant_id === selectedPlantId) || plants[0],
    [plants, selectedPlantId]
  );

  useEffect(() => {
    fetch(`${API_BASE}/plants`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Plant registry unavailable')))
      .then((data) => {
        setPlants(data);
        if (!data.some((plant) => plant.plant_id === selectedPlantId)) {
          setSelectedPlantId(data[0]?.plant_id || 'SOLAR_001');
        }
      })
      .catch(() => setPlants(fallbackPlants));
  }, [selectedPlantId]);

  useEffect(() => {
    let cancelled = false;
    async function loadForecast() {
      setStatus('loading');
      setError('');
      try {
        const forecastResponse = await fetch(`${API_BASE}/forecast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            plant_id: selectedPlantId,
            date: targetDate,
            horizon_hours: Number(horizon),
            forecast_type: horizon <= 6 ? 'intraday' : 'day_ahead',
          }),
        });
        if (!forecastResponse.ok) throw new Error('Forecast API returned an error');
        const forecastData = await forecastResponse.json();
        const actualResponse = await fetch(`${API_BASE}/actuals/${selectedPlantId}?date=${targetDate}&hours=${horizon}`);
        const actualData = actualResponse.ok ? await actualResponse.json() : [];
        if (!cancelled) {
          setForecast(forecastData);
          setActuals(actualData);
          setStatus('ready');
        }
      } catch (loadError) {
        if (!cancelled) {
          setStatus('error');
          setError('Live API is not reachable. Start the backend on port 8000 or use docker-compose up --build.');
        }
      }
    }
    loadForecast();
    return () => {
      cancelled = true;
    };
  }, [selectedPlantId, targetDate, horizon]);

  return (
    <main className="app">
      <section className="hero">
        <div>
          <p className="eyebrow">VASUDHA AI · KREDL / KSPDCL Hackathon</p>
          <h1>Renewable generation forecasts for Karnataka's grid desk.</h1>
          <p className="heroCopy">
            Open-Meteo weather, physics-aware solar and wind models, probabilistic P10/P50/P90 bands, and operator-readable forecast drivers in one live console.
          </p>
        </div>
        <div className="heroBadge">
          <span>Model</span>
          <strong>{forecast?.model_version || 'loading'}</strong>
          <em>{forecast?.weather_source || 'connecting to Open-Meteo'}</em>
        </div>
      </section>

      <section className="toolbar">
        <label>
          Plant
          <select value={selectedPlantId} onChange={(event) => setSelectedPlantId(event.target.value)}>
            {plants.map((plant) => <option key={plant.plant_id} value={plant.plant_id}>{plant.name}</option>)}
          </select>
        </label>
        <label>
          Forecast date
          <input type="date" value={targetDate} onChange={(event) => setTargetDate(event.target.value)} />
        </label>
        <label>
          Horizon
          <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
            <option value={6}>6h intraday</option>
            <option value={24}>24h day-ahead</option>
            <option value={48}>48h outlook</option>
          </select>
        </label>
      </section>

      {error && <div className="errorBanner">{error}</div>}

      <KpiStrip forecast={forecast} />

      <section className="dashboardGrid">
        <PlantMap plants={plants} selectedPlantId={selectedPlantId} onSelect={setSelectedPlantId} />
        <div className="panel assetPanel">
          <p className="eyebrow">Selected asset</p>
          <h2>{selectedPlant?.name}</h2>
          <div className="assetMeta">
            <span>{selectedPlant?.plant_type}</span>
            <span>{selectedPlant?.capacity_mw} MW</span>
            <span>{selectedPlant?.cluster_id}</span>
          </div>
        </div>
      </section>

      <section className="mainGrid">
        {status === 'loading' && <div className="panel loadingPanel">Loading forecast from Open-Meteo...</div>}
        {forecast && status !== 'loading' && <ForecastChart points={forecast.points} actuals={actuals} capacity={forecast.capacity_mw} />}
        {forecast && <ShapChart drivers={forecast.top_drivers || []} />}
      </section>
    </main>
  );
}

export default App;
