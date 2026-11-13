from __future__ import annotations


def radar_map_html(latitude: float, longitude: float) -> str:
    """Build a small Leaflet radar player centered on a validated weather location."""
    latitude = float(latitude)
    longitude = float(longitude)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Radar center coordinates are out of range")
    return f'''<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body{{margin:0;font-family:system-ui,sans-serif;color:#172033}} #map{{height:430px;border-radius:12px}}
.bar{{display:flex;align-items:center;gap:10px;padding:8px 2px;flex-wrap:wrap}} .bar button,.bar select{{padding:7px 10px;border-radius:8px;border:1px solid #bbc4d3;background:white}}
#time{{font-size:13px;color:#42516a}} #status{{font-size:12px;color:#58657a;padding:2px}}
</style></head><body>
<div class="bar"><button id="play">Play</button><input id="frame" type="range" min="0" max="0" value="0" disabled>
<span id="time">Loading radar frames…</span><select id="radius"><option value="5000">Neighborhood view · 5 km</option><option value="25000">City view · 25 km</option></select></div>
<div id="map"></div><div id="status">Radar displays recent observed echoes. It does not show future rainfall.</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const lat={latitude:.6f}, lon={longitude:.6f};
const map=L.map('map').setView([lat,lon],8);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:18,attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);
L.marker([lat,lon]).addTo(map).bindPopup('Selected place');
let ring=L.circle([lat,lon],{{radius:5000,color:'#2563eb',fillOpacity:0.05}}).addTo(map);
document.getElementById('radius').onchange=e=>{{ring.setRadius(Number(e.target.value));map.setZoom(e.target.value==='5000'?11:8)}};
let frames=[], layer=null, timer=null;
const slider=document.getElementById('frame'), label=document.getElementById('time'), play=document.getElementById('play');
function show(i){{if(!frames.length)return;const f=frames[i];if(layer)map.removeLayer(layer);
const url=`${{window.radarHost}}${{f.path}}/256/{{z}}/{{x}}/{{y}}/2/1_0.png`;
layer=L.tileLayer(url,{{opacity:0.7,maxNativeZoom:7,maxZoom:12,errorTileUrl:''}}).addTo(map);
label.textContent=new Date(f.time*1000).toLocaleString()+' (UTC '+new Date(f.time*1000).toISOString().slice(11,16)+')';slider.value=i}}
slider.oninput=()=>show(Number(slider.value));
play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='Play';return}}play.textContent='Pause';timer=setInterval(()=>{{slider.value=(Number(slider.value)+1)%frames.length;show(Number(slider.value))}},800)}};
fetch('https://api.rainviewer.com/public/weather-maps.json').then(r=>r.json()).then(data=>{{
window.radarHost=data.host;frames=(data.radar&&data.radar.past)||[];
if(!frames.length)throw new Error('No recent radar frames for this area');
slider.max=frames.length-1;slider.disabled=false;show(frames.length-1);
}}).catch(()=>{{label.textContent='Radar unavailable';document.getElementById('status').textContent='Radar tiles could not be loaded or coverage is unavailable for this location.'}});
</script></body></html>'''
