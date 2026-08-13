import os
from flask import Flask, render_template_string
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import pandas as pd

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "eda_db"
COLLECTION_NAME = "titanic_cleaned"

app = Flask(__name__, static_folder='visualizations', static_url_path='/visualizations')

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Titanic EDA Dashboard</title>
  <style>
    :root {
      --bg: #050816;
      --surface: rgba(10, 16, 48, 0.94);
      --surface-soft: rgba(15, 23, 54, 0.88);
      --text: #e6edf8;
      --muted: #94a3b8;
      --primary: #60a5fa;
      --accent: #a78bfa;
      --success: #34d399;
      --warning: #facc15;
      --border: rgba(148, 163, 184, 0.18);
      --shadow: 0 32px 80px rgba(7, 12, 30, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
                  radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.14), transparent 24%),
                  linear-gradient(180deg, #06091f 0%, #050816 100%);
    }
    a { color: inherit; text-decoration: none; }
    .page { max-width: 1280px; margin: 0 auto; padding: 0 24px 48px; }
    .site-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 24px 0 12px; }
    .logo { display: inline-flex; align-items: center; gap: 12px; }
    .logo-mark { width: 42px; height: 42px; border-radius: 14px; background: linear-gradient(135deg, #60a5fa, #1d4ed8); display: grid; place-items: center; color: white; font-size: 1.3rem; }
    .logo-name { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.04em; }
    .main-nav { display: flex; align-items: center; gap: 18px; }
    .nav-link { color: var(--text); font-weight: 600; transition: color 0.2s ease; }
    .nav-link:hover { color: #60a5fa; }
    .hero { display: grid; gap: 18px; padding: 46px 0 36px; }
    .hero-title { font-size: clamp(2.8rem, 5vw, 4.6rem); line-height: 0.98; margin: 0; }
    .hero-text { max-width: 760px; color: var(--muted); font-size: 1.05rem; line-height: 1.9; }
    .hero-cta { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 18px; }
    .button { display: inline-flex; align-items: center; justify-content: center; min-width: 160px; padding: 16px 24px; border-radius: 999px; font-weight: 700; transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease; }
    .button.primary { background: linear-gradient(135deg, #60a5fa, #38bdf8); color: white; box-shadow: 0 18px 35px rgba(56, 189, 248, 0.24); }
    .button.secondary { background: rgba(255,255,255,0.08); color: var(--text); border: 1px solid rgba(255,255,255,0.12); }
    .button:hover { transform: translateY(-2px); }
    .visualizations { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 32px; padding: 32px; box-shadow: 0 30px 80px rgba(7, 12, 30, 0.18); }
    .visualizations h2 { margin: 0 0 18px; font-size: clamp(2rem, 3vw, 2.8rem); }
    .chart-grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .chart-card { background: rgba(255,255,255,0.04); padding: 22px; border-radius: 28px; border: 1px solid rgba(255,255,255,0.08); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
    .chart-card h3 { margin: 0 0 16px; font-size: 1.15rem; }
    .chart-card img { width: 100%; height: auto; border-radius: 20px; display: block; }
    .footer { display: flex; justify-content: center; padding: 24px 0 0; color: var(--muted); }
    .ship-overlay { position: fixed; inset: 0; background: radial-gradient(circle at center, rgba(56, 189, 248, 0.18), transparent 28%), rgba(5, 9, 24, 0.98); display: flex; align-items: center; justify-content: center; z-index: 100; flex-direction: column; padding: 24px; transition: opacity 0.3s ease; }
    .ship-overlay.hidden { opacity: 0; visibility: hidden; pointer-events: none; }
    .ship-overlay .ship-circle { width: 260px; height: 260px; border-radius: 50%; background: radial-gradient(circle at top, rgba(96, 165, 250, 0.96), rgba(37, 99, 235, 0.96)); display: flex; align-items: center; justify-content: center; box-shadow: 0 36px 100px rgba(37, 99, 235, 0.35); cursor: pointer; animation: floatShip 4s ease-in-out infinite; }
    .ship-overlay .ship-icon { font-size: 4.5rem; }
    .ship-overlay .overlay-text { margin-top: 22px; max-width: 560px; text-align: center; color: var(--text); font-size: 1.1rem; line-height: 1.8; }
    .hide-non-viz .hero,
    .hide-non-viz .footer { display: none !important; }
    @keyframes floatShip { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-16px); } }
    @media (max-width: 1024px) { .page { padding: 0 18px 36px; } .chart-grid { grid-template-columns: 1fr; } }
    @media (max-width: 720px) { .site-header { flex-direction: column; align-items: flex-start; } .hero-title { font-size: 2.8rem; } }
  </style>
</head>
<body>
  <div id="shipOverlay" class="ship-overlay">
    <div class="ship-circle" id="shipCircle" role="button" aria-label="Open Titanic visualizations">
      <span class="ship-icon">⛴️</span>
    </div>
    <p class="overlay-text">Tap the Titanic to reveal the visual analytics dashboard. Explore the charts and insights on a polished website-style layout.</p>
  </div>

  <div class="page hide-non-viz">
    <header class="site-header">
      <div class="logo">
        <div class="logo-mark">T</div>
        <div class="logo-name">Titanic EDA</div>
      </div>
      <nav class="main-nav">
        <a class="nav-link" href="#visualizations">Visualizations</a>
        <a class="nav-link" href="#visualizations">Charts</a>
      </nav>
    </header>

    <section class="hero">
      <h1 class="hero-title">Titanic Data Visualized as an Elegant Experience</h1>
      <div class="hero-cta">
        <button class="button primary" id="viewCharts">View Visualizations</button>
        <a class="button secondary" href="/visualizations/1_survival_count.png" target="_blank">Open Sample Chart</a>
      </div>
    </section>

    <section id="visualizations" class="visualizations">
      <h2>Visualizations</h2>
      <div class="chart-grid">
        {% for title, filename in plots %}
        <div class="chart-card">
          <h3>{{ title }}</h3>
          <a href="/visualizations/{{ filename }}" target="_blank">
            <img src="/visualizations/{{ filename }}" alt="{{ title }}">
          </a>
        </div>
        {% endfor %}
      </div>
    </section>

    <footer class="footer">
      <p>Flask-powered Titanic analytics. Built as a clean website experience.</p>
    </footer>
  </div>

  <script>
    const shipOverlay = document.getElementById('shipOverlay');
    const page = document.querySelector('.page');
    const shipCircle = document.getElementById('shipCircle');
    const viewCharts = document.getElementById('viewCharts');

    function revealVisualizations() {
      shipOverlay.classList.add('hidden');
      page.classList.remove('hide-non-viz');
      const hero = document.querySelector('.hero');
      hero.scrollIntoView({ behavior: 'smooth' });
    }

    shipCircle.addEventListener('click', revealVisualizations);
    viewCharts.addEventListener('click', revealVisualizations);
  </script>
</body>
</html>
"""

PLOTS = [
    ("Survival Count", "1_survival_count.png"),
    ("Survival Count by Sex", "2_survival_by_sex.png"),
    ("Distribution of Passenger Ages", "3_age_distribution.png"),
    ("Fare Distribution by Passenger Class", "4_fare_by_pclass.png"),
    ("Correlation Heatmap", "5_correlation_heatmap.png"),
]


def get_mongo_client(uri=MONGO_URI):
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def fetch_cleaned_data():
    client = get_mongo_client()
    try:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        records = list(collection.find({}, {'_id': 0}))
        return records
    except ServerSelectionTimeoutError:
        return []
    finally:
        client.close()


def get_summary_stats():
    records = fetch_cleaned_data()
    if not records:
        return {'error': 'No data available. Run eda.py and ensure MongoDB is running.'}
    df = pd.DataFrame(records)
    stats = df.describe(include='all').transpose().to_dict()
    return stats


def get_dashboard_cards(stats):
    try:
        total_passengers = int(stats['survived']['count'])
        survival_rate = round(float(stats['survived']['mean']) * 100, 1)
        avg_age = round(float(stats['age']['mean']), 1)
        avg_fare = round(float(stats['fare']['mean']), 2)
    except (KeyError, TypeError, ValueError):
        return None

    return {
        'total_passengers': total_passengers,
        'survival_rate': survival_rate,
        'avg_age': avg_age,
        'avg_fare': f"{avg_fare:,.2f}",
    }


@app.route('/')
def home():
    stats = get_summary_stats()
    cards = None
    if 'error' not in stats:
        cards = get_dashboard_cards(stats)

    return render_template_string(
        HTML_TEMPLATE,
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        plots=PLOTS,
        stats=stats,
        cards=cards,
    )


@app.route('/api/data')
def api_data():
    return {'data': fetch_cleaned_data()}


@app.route('/api/stats')
def api_stats():
    return {'stats': get_summary_stats()}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
